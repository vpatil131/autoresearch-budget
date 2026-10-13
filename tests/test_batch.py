import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import time
import unittest
from unittest.mock import patch

from autoresearch_budget.batch import BatchRunner, append, coordinator_lock, read_lines
from autoresearch_budget.cli import main
from autoresearch_budget.runner import ExperimentConfig, HarborTrialRunner, RunnerConfigError, TrialSpec


class BatchTest(unittest.TestCase):
    def config(self, root, *, count=8, paid=True):
        tasks = [{"task_id": f"owner__repo-{i}"} for i in range(count)]
        (root / "sample.json").write_text(json.dumps({"tasks": tasks}))
        for task in tasks:
            p = root / "tasks" / task["task_id"]
            p.mkdir(parents=True)
            (p / "task.toml").write_text('[agent]\ntimeout_sec=300\n[environment]\ncpus=1\nmemory_mb=4096\n')
            (p / "instruction.md").write_text("Fix issue " + task["task_id"])
        source = root / "src/autoresearch_budget"
        source.mkdir(parents=True)
        (source / "__init__.py").write_text('"""Offline fixture."""\n')
        (root / ".env").write_text('OPENROUTER_API_KEY=offline-secret\n')
        raw = {
            "budget": {"ledger": "runs/budget.sqlite3", "cap_usd": "15", "enforce": False},
            "locations": {"task_root": "tasks", "jobs_root": "runs/trials"},
            "agent": {"scaffold": "mini-swe-advisor", "version": "2.4.6", "max_steps": 80,
                      "max_tokens_per_call": 4096, "command_timeout_sec": 120, "max_observation_chars": 20000},
            "profiles": {"small": {"model": "deepseek/mock", "run_max_cost_usd": "1", "provider": {"only": ["DeepInfra"]}},
                         "frontier": {"model": "openai/mock", "run_max_cost_usd": "2", "service_tier": "flex", "provider": {"only": ["OpenAI"]}}},
            "paid_inference_enabled": paid,
            "consultation": {"policy": "fixed_interval", "every_steps": 10, "max_calls": 100,
                             "advisor_profile": "frontier", "task_manifest": "sample.json"}}
        path = root / "experiment.json"
        path.write_text(json.dumps(raw))
        return ExperimentConfig.load(path, project_root=root)

    def batch(self, config, **kwargs):
        return BatchRunner(config, config.jobs_root / "_batch/test", **kwargs)

    def outcome(self, claim, *, stop="Submitted", reward=1):
        return {**claim, "return_code": 0, "metrics": {"stop_reason": stop},
                "result": {"verifier_result": {"rewards": {"reward": reward}}}}

    def test_plan_is_read_only_and_cli_plan_does_not_dispatch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); config = self.config(root, paid=False)
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            runner = self.batch(config)
            plan = runner.plan()
            self.assertEqual(len(plan['pending']), 8)
            self.assertEqual(plan['concurrency'], 4)
            self.assertEqual(plan['every_steps'], 10)
            with patch('autoresearch_budget.batch.BatchRunner.run', side_effect=AssertionError('no inference')), patch('pathlib.Path.cwd', return_value=root), patch('builtins.print'):
                self.assertEqual(main(['batch', 'plan', '--config', str(config.path), '--batch-dir', str(runner.batch)]), 0)
            self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))
            with self.assertRaises(RunnerConfigError):runner.run()

    def test_six_concurrent_real_runners_preserve_isolated_logs_and_configuration(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); config = self.config(root, count=12)
            runner = self.batch(config, concurrency=6)
            barrier = threading.Barrier(6)
            mutex = threading.Lock(); running = 0; peak = 0; seen = []
            def harbor(command, **kwargs):
                nonlocal running, peak
                if command[0] == 'git':
                    return type('Result', (), {'returncode': 0, 'stdout': 'offline'})()
                job = command[command.index('--job-name') + 1]
                settings = json.loads(next(a[7:] for a in command if a.startswith('config=')))
                self.assertEqual(settings['agent']['advice']['every_steps'], 10)
                self.assertEqual(settings['agent']['step_limit'], 80)
                self.assertEqual(command[command.index('--n-concurrent') + 1], '1')
                self.assertTrue(kwargs['env']['PYTHONPATH'].startswith(str(runner.batch / 'source')))
                self.assertTrue(Path(kwargs['env']['PYTHONPATH'].split(':')[0], 'autoresearch_budget/__init__.py').exists())
                with mutex:
                    running += 1; peak = max(peak, running); seen.append(job)
                barrier.wait(timeout=5)
                time.sleep(0.005 * (int(job.split('repo-')[1].split('-')[0]) % 6))
                trial = config.jobs_root / job / 'trial'
                (trial / 'agent/advice/artifacts').mkdir(parents=True)
                (trial / 'verifier').mkdir()
                metrics = {'stop_reason': 'Submitted', 'every_steps': 10,
                           'actor': {'api_calls': 21}, 'advisor': {'api_calls': 2}, 'total_cost_usd': '0.01'}
                (trial / 'agent/mini-swe-agent.trajectory.json').write_text(json.dumps({'info': {'poc_metrics': metrics}}))
                (trial / 'agent/advice/events.jsonl').write_text(json.dumps({'run_id': job, 'kind': 'advice_decision'})+'\n')
                (trial / 'agent/advice/artifacts/patch.json').write_text(json.dumps({'run_id': job}))
                (trial / 'verifier/reward.txt').write_text('1')
                (trial / 'result.json').write_text(json.dumps({'finished_at': 'now',
                    'diagnostic': 'offline-secret', 'verifier_result': {'rewards': {'reward': 1}}}))
                kwargs['stdout'].write(job + ' offline-secret\n')
                with mutex:running -= 1
                return type('Result', (), {'returncode': 0})()
            with patch('autoresearch_budget.runner.run_command', side_effect=harbor):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(peak, 6)
            self.assertEqual(len(set(seen)), 12)
            records = read_lines(runner.batch / 'attempts.jsonl')
            self.assertEqual(len(records), 12)
            self.assertNotIn('offline-secret', (runner.batch / 'attempts.jsonl').read_text())
            self.assertEqual(json.loads((runner.batch / 'status.json').read_text())['completed_tasks'], 12)
            for record in records:
                job = record['job_name']; log = Path(record['runner_records']) / 'harbor-console.log'
                self.assertIn(job, log.read_text()); self.assertNotIn('offline-secret', log.read_text())
                self.assertEqual((Path(record['runner_records']) / 'post_submission/trial/verifier/reward.txt').read_text(), '1')
                self.assertTrue((Path(record['job_dir']) / 'trial/agent/advice/artifacts/patch.json').exists())
            events = read_lines(runner.batch / 'events.jsonl')
            self.assertEqual([e['sequence'] for e in events], list(range(1, len(events)+1)))
            with patch.object(runner, 'execute', side_effect=AssertionError('must skip completed tasks')):
                self.assertEqual(runner.run(), 0)
            self.assertEqual(len(read_lines(runner.batch / 'attempts.jsonl')), 12)

    def test_integration_failure_stops_dispatch_but_preserves_other_active_attempts(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp)); runner = self.batch(config)
            started = []; barrier = threading.Barrier(4)
            def execute(claim, source):
                started.append(claim['task_id']); barrier.wait(timeout=5)
                if claim['task_id'].endswith('-0'):return self.outcome(claim, stop='AdvisorUnavailable')
                time.sleep(0.08)
                return self.outcome(claim)
            runner.execute = execute
            self.assertEqual(runner.run(), 2)
            self.assertEqual(len(started), 4)
            status = json.loads((runner.batch/'status.json').read_text())
            self.assertEqual(status['state'], 'needs_attention')
            self.assertEqual(status['completed_tasks'], 3)
            self.assertEqual(len(status['queued_tasks']), 4)
            self.assertEqual(len(read_lines(runner.batch/'attempts.jsonl')), 4)
            with self.assertRaises(RunnerConfigError):runner.run()

    def test_semantic_failures_and_limits_are_valid_attempts(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp), count=5); runner = self.batch(config)
            runner.execute = lambda claim, source: self.outcome(claim, stop='LimitsExceeded', reward=0)
            self.assertEqual(runner.run(), 0)
            self.assertEqual(runner.plan()['completed_tasks'], 5)

    def test_drain_preserves_active_work_and_resume_skips_it(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp)); runner = self.batch(config)
            barrier = threading.Barrier(4)
            def execute(claim, source):
                barrier.wait(timeout=5)
                (runner.batch/'drain.request').touch()
                return self.outcome(claim)
            runner.execute = execute
            self.assertEqual(runner.run(), 0)
            self.assertEqual(json.loads((runner.batch/'status.json').read_text())['state'], 'drained')
            self.assertEqual(len(read_lines(runner.batch/'attempts.jsonl')), 4)
            (runner.batch/'drain.request').unlink()
            runner.execute = lambda claim, source: self.outcome(claim)
            self.assertEqual(runner.run(), 0)
            self.assertEqual(len(read_lines(runner.batch/'attempts.jsonl')), 8)

    def test_recovery_reconciles_terminal_claim_and_blocks_unresolved_claim(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp), count=1); runner = self.batch(config)
            runner.batch.mkdir(parents=True)
            claim = runner._claim(runner.tasks[0], 1)
            append(runner.batch/'claims.jsonl', claim)
            self.assertEqual(len(runner.plan()['unresolved_attempts']), 1)
            with self.assertRaises(RunnerConfigError):runner.run()
            p = Path(claim['runner_records']); p.mkdir(parents=True)
            (p/'summary.json').write_text(json.dumps({'return_code': 0, 'metrics': {'stop_reason': 'Submitted'}}))
            runner.execute = lambda *args: self.fail('must recover instead of retrying')
            self.assertEqual(runner.run(), 0)
            self.assertEqual(len(read_lines(runner.batch/'attempts.jsonl')), 1)

    def test_legacy_running_batch_and_exclusive_lock_block_duplicate_dispatch(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp)); runner = self.batch(config)
            old = config.jobs_root/'_batch/legacy'; old.mkdir(parents=True)
            (old/'status.json').write_text(json.dumps({'state': 'running', 'current_task': runner.tasks[0]}))
            self.assertTrue(runner.plan()['active_batches'])
            with self.assertRaises(RunnerConfigError):runner.run()
            (old/'status.json').unlink()
            with coordinator_lock(config.jobs_root/'_batch/.coordinator.lock'):
                with self.assertRaises(RunnerConfigError):runner.run()

    def test_existing_unindexed_job_and_changed_settings_block_resume(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp), count=1); runner = self.batch(config)
            claim = runner._claim(runner.tasks[0], 1)
            Path(claim['job_dir']).mkdir(parents=True)
            with self.assertRaises(RunnerConfigError):runner.run()
            Path(claim['job_dir']).rmdir()
            runner.execute = lambda claim, source: self.outcome(claim)
            self.assertEqual(runner.run(), 0)
            raw = json.loads(config.path.read_text()); raw['agent']['max_steps'] = 70
            config.path.write_text(json.dumps(raw))
            changed = self.batch(ExperimentConfig.load(config.path, project_root=config.project_root))
            with self.assertRaises(RunnerConfigError):changed.plan()

    def test_invalid_concurrency_and_missing_package_are_rejected(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp))
            for count in (0, 7):
                with self.assertRaises(RunnerConfigError):self.batch(config, concurrency=count)
            (config.task_root/'owner__repo-0/instruction.md').unlink()
            with self.assertRaises(RunnerConfigError):self.batch(config)

    def test_phase3_modes_have_distinct_jobs_frozen_configs_and_resume_guards(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); config = self.config(root, count=1)
            names = []
            for mode in ('phase3_control', 'plan_review', 'recovery_review', 'hybrid'):
                batch_dir = config.jobs_root/'_batch'/mode
                argv = ['batch', 'run', '--config', str(config.path), '--batch-dir', str(batch_dir),
                        '--advice-policy', mode, '--max-advice-calls', '7']
                def execute(runner, claim, source):
                    settings = json.loads(next(a[7:] for a in runner.runner.command(
                        TrialSpec(claim['task_id'], 'small')) if a.startswith('config=')))
                    self.assertEqual(settings['agent']['advice']['policy'], mode)
                    names.append(claim['job_name'])
                    return self.outcome(claim)
                with patch('pathlib.Path.cwd', return_value=root), patch.object(BatchRunner, '_execute', execute):
                    self.assertEqual(main(argv), 0)
                    self.assertEqual(main(argv), 0)
                    with patch('builtins.print'):
                        self.assertEqual(main([*argv[:-1], '6']), 2)
                frozen = ExperimentConfig.load(batch_dir/'config.json', project_root=root)
                self.assertEqual(frozen.consultation['policy'], mode)
            self.assertEqual(len(set(names)), 4)
            self.assertTrue(all(name.startswith('phase3-') for name in names))

    def test_requested_policy_cli_overrides_freeze_and_resume_exact_settings(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); config=self.config(root,count=1)
            batch_dir=config.jobs_root/'_batch/requested'
            argv=['batch','run','--config',str(config.path),'--batch-dir',str(batch_dir),
                  '--advice-policy','advice_requested','--max-advice-calls','3']
            seen=[]
            def execute(runner,claim,source):
                frozen=ExperimentConfig.load(batch_dir/'config.json',project_root=root)
                settings=json.loads(next(a[7:] for a in runner.runner.command(
                    TrialSpec(claim['task_id'],'small'))
                    if a.startswith('config=')))
                seen.append(settings['agent']['advice'])
                self.assertEqual(frozen.consultation['policy'],'advice_requested')
                self.assertEqual(frozen.consultation['max_calls'],3)
                self.assertNotIn('every_steps',frozen.consultation)
                return self.outcome(claim)
            with patch('pathlib.Path.cwd',return_value=root), patch.object(BatchRunner,'_execute',execute):
                self.assertEqual(main(argv),0)
                self.assertEqual(main(argv),0)
                with patch('builtins.print'):
                    self.assertEqual(main([*argv[:-1],'4']),2)
            self.assertEqual(len(seen),1)
            self.assertEqual(seen[0]['max_calls'],3)
            records=read_lines(batch_dir/'attempts.jsonl')
            self.assertTrue(records[0]['job_name'].startswith('phase2-advice_requested-m3-'))
            metadata=json.loads((batch_dir/'metadata.json').read_text())
            self.assertEqual(metadata['config']['consultation']['max_calls'],3)
            self.assertEqual(json.loads(config.path.read_text())['consultation']['policy'],'fixed_interval')

    def test_requested_and_actor_only_batch_plans_and_invalid_overrides_are_read_only(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); config=self.config(root,count=1,paid=False)
            for policy in ('advice_requested','actor_only'):
                resolved=config.with_consultation_overrides(policy=policy,max_calls=0)
                plan=self.batch(resolved).plan()
                self.assertEqual(plan['policy'],policy)
                self.assertEqual(plan['max_advice_calls'],0)
                self.assertIsNone(plan['every_steps'])
            for flags in (['--max-advice-calls','-1'],
                          ['--advice-policy','advice_requested','--advice-every-steps','10']):
                with patch('pathlib.Path.cwd',return_value=root), patch('builtins.print'), \
                     patch.object(BatchRunner,'run',side_effect=AssertionError('no dispatch')):
                    self.assertEqual(main(['batch','run','--config',str(config.path),'--batch-dir',str(root/'batch'),*flags]),2)
            self.assertFalse((root/'batch').exists())

    def test_zero_ceiling_cli_retains_hybrid_without_requiring_advisor_profile(self):
        with TemporaryDirectory() as temp:
            root = Path(temp); config = self.config(root, count=1, paid=False)
            raw = json.loads(config.path.read_text())
            del raw['profiles']['frontier']
            del raw['consultation']['advisor_profile']
            config.path.write_text(json.dumps(raw))
            argv = ['batch', 'plan', '--config', str(config.path), '--batch-dir', str(root/'batch'),
                    '--advice-policy', 'hybrid', '--max-advice-calls', '0']
            with patch('pathlib.Path.cwd', return_value=root), patch('builtins.print') as printed:
                self.assertEqual(main(argv), 0)
            preview = json.loads(printed.call_args.args[0])
            self.assertFalse(preview['advice_enabled'])
            self.assertTrue(preview['planning_enabled'])
            self.assertTrue(preview['recovery_enabled'])
            self.assertEqual(set(preview['models']), {'actor'})
            resolved = ExperimentConfig.load(config.path, project_root=root).with_consultation_overrides(
                policy='hybrid', max_calls=0)
            settings = json.loads(next(a[7:] for a in self.batch(resolved).runner.command(
                TrialSpec('owner__repo-0', 'small')) if a.startswith('config=')))
            self.assertEqual(settings['agent']['advice']['advisor'], {})
            self.assertFalse((root/'batch').exists())
            # Re-enabling advice must still validate the advisor route.
            with self.assertRaises(RunnerConfigError):
                self.batch(resolved.with_consultation_overrides(max_calls=1))

    def test_worker_exception_is_journaled_without_exposing_exception_payload(self):
        with TemporaryDirectory() as temp:
            config = self.config(Path(temp), count=1); runner = self.batch(config)
            def execute(*args):raise ValueError('offline-secret payload')
            runner.execute = execute
            self.assertEqual(runner.run(), 2)
            records = read_lines(runner.batch/'attempts.jsonl')
            self.assertEqual(records[0]['error'], 'ValueError')
            self.assertNotIn('offline-secret', (runner.batch/'status.json').read_text())

    def test_resume_rejects_changed_task_contents_or_runtime(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);config=self.config(root,count=1);runner=self.batch(config)
            runner.execute=lambda claim,source:self.outcome(claim)
            self.assertEqual(runner.run(),0)
            instruction=config.task_root/runner.tasks[0]/'instruction.md'
            original=instruction.read_text();instruction.write_text('Changed issue')
            with self.assertRaisesRegex(RunnerConfigError,'task contents or runtime changed'):
                self.batch(config).plan()
            instruction.write_text(original)
            with patch('autoresearch_budget.batch.runtime_identity',return_value={'source_sha256':{'changed.py':'hash'}}):
                with self.assertRaisesRegex(RunnerConfigError,'task contents or runtime changed'):
                    self.batch(config).plan()

    def test_worker_uses_the_frozen_manifest_instead_of_mutable_workspace_copy(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary);config=self.config(root,count=1);runner=self.batch(config)
            runner.batch.mkdir(parents=True)
            (runner.batch/'config.json').write_text(json.dumps(config.resolved_config()))
            (runner.batch/'manifest.json').write_text(json.dumps(runner.manifest))
            (root/'sample.json').write_text('{"tasks":[]}')
            def execute(instance,spec,**kwargs):
                self.assertEqual(instance.config.consultation['task_manifest'],str(runner.batch/'manifest.json'))
                self.assertEqual(instance.config.consultation['policy'],'fixed_interval')
                return 0
            with patch.object(HarborTrialRunner,'run',autospec=True,side_effect=execute):
                result=runner._execute(runner._claim(runner.tasks[0],1),root/'source')
            self.assertEqual(result['return_code'],0)


if __name__ == '__main__':unittest.main()
