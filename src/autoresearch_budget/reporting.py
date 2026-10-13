"""Read-only summaries and descriptive matched comparisons of local artifacts."""
from __future__ import annotations

import csv
from datetime import datetime
from decimal import Decimal, InvalidOperation
import io
import json
import os
from pathlib import Path

from .advisor import actor_usage
from .runner import _completed_job_cost

SCHEMA_VERSION = 1
VALID_STOPS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}


def _read(path: Path, errors: list) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError('expected JSON object')
        return value
    except (OSError, ValueError) as error:
        errors.append({"file": path.name, "error": type(error).__name__})
        return {}


def _amount(value):
    try:
        amount = Decimal(str(value))
        return amount if amount.is_finite() and amount >= 0 else None
    except (InvalidOperation, ValueError):
        return None


def _duration(value):
    if not isinstance(value, dict) or not value.get('started_at') or not value.get('finished_at'):
        return None
    try:
        seconds = (datetime.fromisoformat(value['finished_at']) - datetime.fromisoformat(value['started_at'])).total_seconds()
        return seconds if seconds >= 0 else None
    except (TypeError, ValueError):
        return None


def _attempt(job: Path, records: Path | None, base: Path, claim: dict | None = None) -> dict:
    errors = []
    metadata = _read(records/'metadata.json', errors) if records else {}
    summary = _read(records/'summary.json', errors) if records else {}
    job_result = _read(job/'result.json', errors)
    paths = sorted(job.glob('*/result.json'))
    result = _read(paths[0], errors) if len(paths) == 1 else (claim or {}).get('result', {})
    if len(paths) > 1:
        errors.append({'file': 'result.json', 'error': 'multiple_trials'})
    trajectories = sorted(job.glob('*/agent/mini-swe-agent.trajectory.json'))
    if (job/'agent/mini-swe-agent.trajectory.json').is_file():
        trajectories = [job/'agent/mini-swe-agent.trajectory.json']
    trajectory = _read(trajectories[0], errors) if len(trajectories) == 1 else {}
    info = trajectory.get('info') or {}
    metrics = summary.get('metrics') or info.get('poc_metrics') or (claim or {}).get('metrics') or {}
    actor = metrics.get('actor') or (actor_usage(trajectory.get('messages', []), info.get('model_stats', {}).get('api_calls', 0)) if trajectory and 'api_calls' in (info.get('model_stats') or {}) else {})
    advisor = metrics.get('advisor') or {}
    stop = metrics.get('stop_reason') or info.get('exit_status')
    if metadata.get('kind') == 'scripted-demo':
        result = {}  # A scripted public assertion is not a benchmark verifier.
    rewards = (result.get('verifier_result') or {}).get('rewards') or {}
    reward = _amount(rewards.get('reward'))
    exception = (result.get('exception_info') or {}).get('exception_type') or summary.get('exception_type')
    return_code = summary.get('return_code', (claim or {}).get('return_code'))
    failed = return_code not in (None, 0) or bool(exception) or (stop is not None and stop not in VALID_STOPS)
    status = 'error' if failed else 'completed' if summary.get('status') == 'completed' or result.get('finished_at') else 'incomplete'
    if metrics.get('total_cost_usd') is None and not metrics and summary.get('status') == 'interrupted_or_runner_error':
        status = 'incomplete'
    if errors:
        status = 'incomplete_records'
    if metrics:
        cost = _amount(metrics.get('total_cost_usd'))
    elif actor:
        cost = _amount(actor.get('cost_usd'))
    else:
        cost = _amount(summary.get('total_cost_usd'))
        if cost is None:
            cost = _completed_job_cost(job/'result.json')
    observed = _amount(actor.get('observed_cost_usd'))
    observed_advisor = _amount(advisor.get('observed_cost_usd'))
    observed = (observed or Decimal(0)) + (observed_advisor or Decimal(0)) if actor or advisor else cost
    configuration = metadata.get('configuration') or {}
    task = metadata.get('task_id') or (claim or {}).get('task_id') or result.get('task_name')
    links = {}
    def link(name, path):
        if path and path.exists():
            links[name] = os.path.relpath(path, base)
    link('metadata', records/'metadata.json' if records else None)
    link('summary', records/'summary.json' if records else None)
    link('job_result', job/'result.json')
    link('trial_result', paths[0] if len(paths) == 1 else None)
    link('trajectory', trajectories[0] if len(trajectories) == 1 else None)
    link('patch', job/'patch.diff')
    if len(trajectories) == 1:
        agent_dir = trajectories[0].parent
        link('events', agent_dir/'advice/events.jsonl')
        link('advice_artifacts', agent_dir/'advice/artifacts')
        link('verifier', agent_dir.parent/'verifier')
        for candidate in ('patch.diff', 'patch.patch'):
            link('patch', agent_dir/candidate)
    provenance = metadata.get('provenance') or {}
    return {'task_id': task, 'profile': metadata.get('profile') or (claim or {}).get('profile'),
            'attempt': metadata.get('attempt') or (claim or {}).get('attempt'), 'job_name': job.name,
            'status': status, 'return_code': return_code, 'stop_reason': stop,
            'failure_category': 'runner' if summary.get('exception_type') else 'provider_or_agent' if stop and stop not in VALID_STOPS else 'infrastructure_or_verifier' if exception else 'runner' if failed else None,
            'exception_type': exception, 'grading_status': 'error' if exception else 'completed' if reward is not None else 'unavailable',
            'reward': float(reward) if reward is not None else None,
            'outcome': 'passed' if reward == 1 else 'failed' if reward is not None else 'unknown',
            'budget_enforced': metadata.get('budget_enforced', summary.get('budget_enforced')),
            'actor': actor, 'advisor': advisor, 'total_cost_usd': str(cost) if cost is not None else None,
            'observed_cost_usd': str(observed) if observed is not None else None,
            'unknown_actor_cost_calls': max(0, actor.get('api_calls', 0)-actor.get('responses_with_cost', actor.get('api_calls', 0))) if actor else None,
            'unknown_advisor_cost_calls': advisor.get('unknown_cost_calls'),
            'duration_seconds': summary.get('duration_seconds'), 'agent_seconds': _duration(result.get('agent_execution')) or metrics.get('duration_seconds'),
            'setup_seconds': _duration(result.get('environment_setup')), 'grading_seconds': _duration(result.get('verifier')),
            'end_to_end_seconds': _duration(result), 'configuration': configuration, 'provenance': provenance,
            'evidence': links, 'record_errors': errors}


def report(source: Path, *, link_root: Path | None = None) -> dict:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise ValueError('report source must be a run, jobs directory or batch directory')
    base = (link_root or source).resolve()
    errors = []
    rows = []
    wall = None
    if (source/'claims.jsonl').exists() or (source/'attempts.jsonl').exists():
        metadata = _read(source/'metadata.json', errors)
        root = (source/metadata.get('jobs_root_relative', '../..')).resolve()
        claims = {}
        for filename in ('claims.jsonl', 'attempts.jsonl'):
            path = source/filename
            if path.is_file():
                for index, line in enumerate(path.read_text().splitlines(), 1):
                    try:
                        record = json.loads(line)
                        claims[record['job_name']] = record
                    except (ValueError, KeyError, TypeError):
                        errors.append({'file': filename, 'line': index, 'error': 'invalid_record'})
        for name, claim in sorted(claims.items()):
            if not isinstance(name, str) or Path(name).name != name or name in {'.', '..'}:
                raise ValueError('invalid job name in batch journal')
            records = root/'_poc_records'/name
            rows.append(_attempt(root/name, records, base, claim))
        state = _read(source/'status.json', errors)
        wall = _duration({'started_at': metadata.get('started_at'), 'finished_at': state.get('updated_at')}) if state.get('state') in {'complete', 'drained', 'needs_attention'} else None
    elif (source/'metadata.json').is_file() and source.parent.name in {'_run_records', '_poc_records'}:
        rows.append(_attempt(source.parent.parent/source.name, source, base))
    elif (source/'agent/mini-swe-agent.trajectory.json').is_file():
        rows.append(_attempt(source, source, base))
    elif (source/'result.json').is_file():
        records = next((source.parent/kind/source.name for kind in ('_run_records','_poc_records') if (source.parent/kind/source.name).is_dir()),None)
        rows.append(_attempt(source, records, base))
    else:
        names = set()
        for kind in ('_run_records', '_poc_records'):
            for directory in (source/kind).glob('*'):
                if directory.is_dir():
                    names.add(directory.name)
                    rows.append(_attempt(source/directory.name, directory, base))
        for directory in sorted(source.iterdir()):
            if directory.is_dir() and not directory.name.startswith('_') and directory.name not in names and (directory/'result.json').is_file():
                rows.append(_attempt(directory, None, base))
        if not rows and ((source/'result.json').is_file() or list(source.glob('*/agent/mini-swe-agent.trajectory.json'))):
            records = next((source.parent/kind/source.name for kind in ('_run_records','_poc_records') if (source.parent/kind/source.name).is_dir()),None)
            rows.append(_attempt(source, records, base))
    if not rows:
        raise ValueError('no supported attempt artifacts found')
    known = sum((Decimal(r['total_cost_usd']) for r in rows if r['total_cost_usd'] is not None), Decimal(0))
    unknown = sum(r['total_cost_usd'] is None for r in rows)
    return {'schema_version': SCHEMA_VERSION, 'evidence_base': '.', 'attempts': rows, 'record_errors': errors,
            'summary': {'attempts': len(rows), 'passed': sum(r['outcome']=='passed' for r in rows),
                        'failed': sum(r['outcome']=='failed' for r in rows), 'ungraded': sum(r['outcome']=='unknown' for r in rows),
                        'known_complete_attempt_cost_usd': str(known), 'unknown_cost_attempts': unknown,
                        'total_cost_usd': str(known) if not unknown else None,
                        'summed_task_duration_seconds': sum(r['duration_seconds'] for r in rows if r['duration_seconds'] is not None) if any(r['duration_seconds'] is not None for r in rows) else None,
                        'known_duration_attempts': sum(r['duration_seconds'] is not None for r in rows),
                        'batch_wall_seconds': wall}}


def compare(left: dict, right: dict) -> dict:
    def index(report):
        rows = {}
        for row in report['attempts']:
            task = row['task_id']
            if not task:
                raise ValueError('comparison needs recorded task identities')
            if task in rows:
                raise ValueError(f'multiple attempts for {task}; select one attempt per condition')
            rows[task] = row
        return rows
    old, new = index(left), index(right)
    pairs = []
    for task in sorted(old.keys() & new.keys()):
        a,b=old[task],new[task]
        issues=[]
        for field in ('task_package_sha256','runtime'):
            x,y=a['provenance'].get(field),b['provenance'].get(field)
            if not x or not y:issues.append(f'missing {field}')
            elif x != y:issues.append(f'different {field}')
        eligible = not issues and a['grading_status']=='completed' and b['grading_status']=='completed' and a['status']=='completed' and b['status']=='completed'
        delta = 'gain' if a['outcome']=='failed' and b['outcome']=='passed' else 'regression' if a['outcome']=='passed' and b['outcome']=='failed' else 'shared_pass' if a['outcome']==b['outcome']=='passed' else 'shared_failure' if a['outcome']==b['outcome']=='failed' else 'ungraded'
        differences = [key for key in ('agent','profiles','consultation','budget') if a['configuration'].get(key) != b['configuration'].get(key)]
        pairs.append({'task_id':task,'left_reward':a['reward'],'right_reward':b['reward'],'change':delta,
                      'eligible_for_aggregate':eligible,'provenance_issues':issues,'configuration_differences':differences})
    eligible = [p for p in pairs if p['eligible_for_aggregate']]
    return {'schema_version':SCHEMA_VERSION,'method':'Descriptive matched-task comparison; configuration differences do not establish causality.',
            'left_only': sorted(old.keys()-new.keys()),'right_only': sorted(new.keys()-old.keys()),'pairs':pairs,
            'summary':{'matched_tasks':len(pairs),'eligible_scored_pairs':len(eligible),
                       **{label:sum(p['change']==label for p in eligible) for label in ('gain','regression','shared_pass','shared_failure')}}}


def render(result: dict, format: str) -> str:
    if format == 'json':
        return json.dumps(result, indent=2) + '\n'
    if 'pairs' in result:
        return json.dumps(result, indent=2) + '\n'
    if format == 'csv':
        output=io.StringIO(); fields=('task_id','profile','attempt','status','grading_status','outcome','reward','stop_reason','total_cost_usd','duration_seconds')
        writer=csv.DictWriter(output,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['attempts']);return output.getvalue()
    summary=result['summary']
    lines=[f"{summary['attempts']} attempts: {summary['passed']} passed, {summary['failed']} failed, {summary['ungraded']} ungraded",
           f"Known complete-attempt costs: ${summary['known_complete_attempt_cost_usd']}; {summary['unknown_cost_attempts']} attempts have unknown costs."]
    lines += [f"{r['task_id'] or r['job_name']}: {r['outcome']} ({r['status']}); cost={r['total_cost_usd'] if r['total_cost_usd'] is not None else 'unknown'}" for r in result['attempts']]
    return '\n'.join(lines)+'\n'
