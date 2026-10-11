import contextlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
from unittest.mock import patch

from autoresearch_budget.budget import BudgetLedger
from autoresearch_budget.cli import main
from autoresearch_budget.runner import ExperimentConfig, HarborTrialRunner, RunnerConfigError, TrialSpec
from autoresearch_budget.trial_pool import TrialPool, select_tasks


class TrialPoolTest(unittest.TestCase):
    def config(self, root, count=3, paid=True):
        tasks = [f"owner__repo-{i}" for i in range(count)]
        for task in tasks:
            directory = root / "tasks" / task
            directory.mkdir(parents=True)
            (directory / "task.toml").write_text('[environment]\ncpus=1\n')
            (directory / "instruction.md").write_text("Fix the issue")
        (root / "sample.json").write_text(json.dumps({"tasks": [{"task_id": task} for task in tasks]}))
        config = {"budget": {"ledger": "runs/ledger.sqlite3", "cap_usd": "15"},
                  "locations": {"task_root": "tasks", "jobs_root": "runs/trials"},
                  "agent": {"scaffold": "mini-swe-agent", "version": "2.4.6", "max_steps": 80,
                            "command_timeout_sec": 120, "max_tokens_per_call": 4096,
                            "max_observation_chars": 20000},
                  "profiles": {"small": {"model": "mock/actor", "run_max_cost_usd": "0.75"}},
                  "paid_inference_enabled": paid}
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return ExperimentConfig.load(path, project_root=root), [TrialSpec(task, "small") for task in tasks]

    def test_eight_workers_execute_distinct_trials_and_share_budget_safely(self):
        with TemporaryDirectory() as temp:
            config, specs = self.config(Path(temp), count=16)
            runner = HarborTrialRunner(config)
            barrier, lock = threading.Barrier(8), threading.Lock()
            active = peak = 0
            seen, reports = [], []

            def execute(spec, *, env_file):
                nonlocal active, peak
                self.assertEqual(env_file, "custom.env")
                ledger = BudgetLedger(config.budget_db, config.budget_cap_usd)
                ledger.reserve(spec.task_id, run_id=spec.task_id, model="mock/actor", max_cost_usd="0.75")
                with lock:
                    active += 1
                    peak = max(peak, active)
                    seen.append(spec.task_id)
                barrier.wait(timeout=5)
                ledger.reconcile(spec.task_id, "0.01")
                with lock:
                    active -= 1
                return 0

            pool = TrialPool(runner, specs, workers=8, env_file="custom.env")
            with patch.object(runner, "run", side_effect=execute):
                self.assertEqual(pool.run(reports.append), 0)
            self.assertEqual(peak, 8)
            self.assertEqual(sorted(seen), sorted(spec.task_id for spec in specs))
            self.assertEqual(len(reports), 16)
            self.assertTrue(all(record["return_code"] == 0 for record in reports))
            ledger = BudgetLedger(config.budget_db, config.budget_cap_usd)
            self.assertEqual(str(ledger.snapshot().settled_usd), "0.160000")
            self.assertEqual(ledger.snapshot().held_usd, 0)

    def test_failure_stops_new_dispatch_and_drains_active_trials(self):
        with TemporaryDirectory() as temp:
            config, specs = self.config(Path(temp), count=5)
            runner = HarborTrialRunner(config)
            barrier, failure_reported = threading.Barrier(2), threading.Event()
            seen, reports = [], []

            def execute(spec, **kwargs):
                seen.append(spec.task_id)
                barrier.wait(timeout=5)
                if spec == specs[0]:
                    raise RunnerConfigError("offline startup failure")
                self.assertTrue(failure_reported.wait(timeout=5))
                return 0

            def report(record):
                reports.append(record)
                if record["return_code"]:
                    failure_reported.set()

            with patch.object(runner, "run", side_effect=execute):
                self.assertEqual(TrialPool(runner, specs, workers=2).run(report), 1)
            self.assertEqual(set(seen), {spec.task_id for spec in specs[:2]})
            self.assertEqual(len(reports), 2)
            self.assertIn("offline startup failure", reports[0]["error"])

    def test_invalid_workers_and_duplicates_cannot_dispatch(self):
        with TemporaryDirectory() as temp:
            config, specs = self.config(Path(temp))
            runner = HarborTrialRunner(config)
            for workers in (-1, 0, 9, 100):
                with self.subTest(workers=workers), self.assertRaises(RunnerConfigError):
                    TrialPool(runner, specs, workers=workers)
            with self.assertRaises(RunnerConfigError):
                TrialPool(runner, [specs[0], specs[0]], workers=2)
            with self.assertRaises(RunnerConfigError):
                select_tasks(["", "valid"], None)
            with self.assertRaises(RunnerConfigError):
                select_tasks(["same", "same"], None)

    def test_all_packages_and_attempts_preflight_before_execution(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config, specs = self.config(root)
            runner = HarborTrialRunner(config)
            pool = TrialPool(runner, specs, workers=3)
            last_instruction = config.task_root / specs[-1].task_id / "instruction.md"
            last_instruction.unlink()
            with patch.object(runner, "run") as execute:
                with self.assertRaisesRegex(RunnerConfigError, "incomplete"):
                    pool.run()
                execute.assert_not_called()
            last_instruction.write_text("Fix the issue")
            (config.jobs_root / runner._job_name(specs[-1])).mkdir(parents=True)
            with patch.object(runner, "run") as execute:
                with self.assertRaisesRegex(RunnerConfigError, "already exists"):
                    pool.run()
                execute.assert_not_called()

    def test_disabled_execution_and_unknown_profile_fail_before_dispatch(self):
        with TemporaryDirectory() as temp:
            config, specs = self.config(Path(temp), paid=False)
            runner = HarborTrialRunner(config)
            with patch.object(runner, "run") as execute:
                with self.assertRaises(RunnerConfigError):
                    TrialPool(runner, specs, workers=2).run()
                execute.assert_not_called()
            with self.assertRaisesRegex(RunnerConfigError, "unknown model profile"):
                TrialPool(runner, [*specs, TrialSpec("other", "missing")])

    def test_cli_manifest_plan_is_read_only_and_single_task_stays_compatible(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config, specs = self.config(root)
            before = sorted(str(p.relative_to(root)) for p in root.rglob('*'))
            with patch('pathlib.Path.cwd', return_value=root), patch.object(HarborTrialRunner, 'run') as execute:
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(main(['trial', 'plan', '--config', str(config.path), '--profile', 'small',
                        '--task-manifest', str(root / 'sample.json'), '--workers', '8']), 0)
                plan = json.loads(output.getvalue())
                self.assertEqual(plan['workers'], 8)
                self.assertEqual(plan['effective_workers'], 3)
                self.assertEqual(len(plan['trials']), 3)
                execute.assert_not_called()
                self.assertEqual(before, sorted(str(p.relative_to(root)) for p in root.rglob('*')))
                execute.return_value = 7
                self.assertEqual(main(['trial', 'run', '--config', str(config.path), '--profile', 'small',
                    '--task-id', specs[0].task_id]), 7)
                execute.assert_called_once_with(specs[0], env_file='.env')

    def test_cli_repeat_task_ids_and_worker_limit(self):
        with TemporaryDirectory() as temp:
            config, specs = self.config(Path(temp))
            base = ['trial', 'run', '--config', str(config.path), '--profile', 'small']
            with patch('pathlib.Path.cwd', return_value=config.project_root), patch.object(TrialPool, 'run', return_value=0) as execute:
                self.assertEqual(main([*base, '--task-id', specs[0].task_id,
                    '--task-id', specs[1].task_id, '--workers', '4']), 0)
                execute.assert_called_once()
                execute.reset_mock()
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                    main([*base, '--task-id', specs[0].task_id, '--workers', '9'])
                self.assertEqual(raised.exception.code, 2)
                execute.assert_not_called()

    def test_conflicting_existing_budget_cannot_dispatch(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            config, specs = self.config(root)
            BudgetLedger(config.budget_db, "1")
            with patch('pathlib.Path.cwd', return_value=root), patch.object(HarborTrialRunner, 'run') as execute:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(main(['trial', 'run', '--config', str(config.path), '--profile', 'small',
                        '--task-manifest', str(root / 'sample.json'), '--workers', '8']), 2)
                self.assertIn('different cap', output.getvalue())
                execute.assert_not_called()
