"""Read-only snapshots of every active advisor task, including live containers."""
from collections import Counter
import inspect
import json
from pathlib import Path
import subprocess

from .batch import read_json


def summarize(summary, events):
    counts = Counter()
    routes = Counter()
    checkpoints = []
    decisions = []
    requests = []
    interventions = []
    recovery_checkpoints = []
    inspection_checkpoints = []
    plans = []
    denials = Counter()
    latest = None
    contiguous = True
    for index, line in enumerate(events, 1):
        try:
            event = json.loads(line)
        except ValueError:
            # An append may still be in flight; report it without inventing a stall.
            contiguous = False
            continue
        counts[event['kind']] += 1
        contiguous = contiguous and event.get('sequence') == index
        latest = {k: event.get(k) for k in ('kind', 'timestamp', 'sequence')}
        payload = event.get('payload', {})
        if event['kind'] == 'advice_requested':
            if payload.get('reason', 'fixed_interval') == 'fixed_interval':
                checkpoints.append(payload.get('actor_call'))
        if event['kind'] == 'intervention':
            interventions.append(payload)
        if event['kind'] == 'recovery_checkpoint':
            recovery_checkpoints.append(payload)
        if event['kind'] == 'inspection_checkpoint':
            inspection_checkpoints.append(payload)
        if event['kind'] == 'plan_proposed':
            plans.append(payload)
        if event['kind'] == 'actor_advice_request':
            requests.append({k: payload.get(k) for k in ('actor_call', 'question', 'evidence_ids')})
        if event['kind'] == 'advice_denied':
            denials[payload.get('reason')] += 1
        if event['kind'] == 'advice_decision':
            decisions.append({k: payload.get(k) for k in ('actor_call', 'advice_id', 'decision')})
        if event['kind'] == 'model_response':
            body = payload.get('raw_body', {})
            if isinstance(body, str):
                try:
                    body = json.loads(body)
                except ValueError:
                    body = {}
            routes[(payload.get('model_role'), body.get('provider'), body.get('service_tier'))] += 1
    ceiling = summary.get('max_advice_calls')
    used = counts['advice_requested']
    return {'summary': summary, 'latest_activity': latest, 'checkpoints': checkpoints,
            'advice_requests': requests, 'advice_dispatched': used,
            'interventions': interventions, 'recovery_checkpoints': recovery_checkpoints,
            'inspection_checkpoints': inspection_checkpoints, 'plans': plans,
            'advice_delivered': counts['advice_delivered'], 'advice_denials': dict(denials),
            'advice_calls_remaining': max(0, ceiling - used) if ceiling is not None else None,
            'decisions': decisions, 'event_sequence_contiguous': contiguous,
            'routes': [{'role': role, 'provider': provider, 'service_tier': tier, 'responses': n}
                       for (role, provider, tier), n in routes.items()],
            'errors': {kind: counts[kind] for kind in ('actor_schema_error', 'advice_schema_error',
                       'advice_failed', 'model_transport_error', 'patch_snapshot_failed')},
            'patch_snapshots': counts['patch_snapshot']}


def snapshot(batch_dir: Path, *, live=False) -> dict:
    batch_dir = batch_dir.resolve()
    status = read_json(batch_dir/'status.json', {})
    active = list(status.get('active_tasks', []))
    # Continue to display the original sequential status during migration.
    if status.get('current_task'):
        records = read_json(batch_dir/'config.json', {})
        root = Path(records.get('locations', {}).get('jobs_root', 'runs/phase1-trials')).resolve()
        jobs = sorted(root.glob('phase1-*-'+status['current_task']+'-a*'), key=lambda p: p.stat().st_mtime)
        if jobs:
            active.append({'task_id': status['current_task'], 'job_name': jobs[-1].name, 'job_dir': str(jobs[-1])})
    containers = []
    docker_error = None
    if live and active:
        try:
            ps = subprocess.run(['docker', 'ps', '--format', '{{json .}}'], capture_output=True,
                                text=True, timeout=10, check=True)
            containers = [json.loads(line) for line in ps.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            docker_error = type(error).__name__
    progress = []
    for claim in active:
        info = {'task_id': claim['task_id'], 'job_name': claim['job_name'],
                'job_dir': claim['job_dir'], 'source': 'unavailable'}
        trials = list(Path(claim['job_dir']).glob('*/agent/advice'))
        if len(trials) == 1:
            directory = trials[0]
            try:
                summary = read_json(directory/'summary.json', {})
                if (directory/'events.jsonl').is_file():
                    with (directory/'events.jsonl').open() as stream:
                        info.update(summarize(summary, stream))
                    info['source'] = 'host_records'
            except (OSError, ValueError, KeyError) as error:
                info['host_error'] = type(error).__name__
        if live:
            trial_names = {p.name.lower() for p in Path(claim['job_dir']).iterdir() if p.is_dir()} if Path(claim['job_dir']).exists() else set()
            matches = [c for c in containers if c.get('Names') in {n+'__env-main-1' for n in trial_names}]
            if len(matches) == 1:
                # The adapter installs this symlink on every task. Resolve its Python
                # without assuming HOME/XDG tool paths or invoking a model process.
                # batch_monitor is a host utility, not installed in the solving package:
                # embed only the read-only summarizer in the inspection program.
                program = ('import json; from collections import Counter; from pathlib import Path;\n'
                           + inspect.getsource(summarize) + '\n'
                           'p=Path("/logs/agent/advice")\n'
                           'with (p/"events.jsonl").open() as f:\n'
                           ' print(json.dumps(summarize(json.loads((p/"summary.json").read_text()),f)))\n')
                command = ['docker', 'exec', matches[0]['ID'], 'sh', '-c',
                           'exec "$(dirname "$(readlink -f "$HOME/.local/bin/mini-swe-agent")")/python" -c "$1"',
                           'read-advice', program]
                try:
                    result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=True)
                    info.update(json.loads(result.stdout)); info['source'] = 'live_container'
                    info['container_id'] = matches[0]['ID']
                except (OSError, subprocess.SubprocessError, ValueError) as error:
                    info['live_error'] = type(error).__name__
            else:
                info['live_error'] = docker_error or 'NoUniqueRunningContainer'
        progress.append(info)
    return {'batch': status, 'progress': progress}
