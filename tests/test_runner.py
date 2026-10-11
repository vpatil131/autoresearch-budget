import json
import os
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autoresearch_budget.runner import (
    ExperimentConfig,
    HarborTrialRunner,
    RunnerConfigError,
    TrialSpec,
    _completed_job_cost,
    _read_env_file,
)
from autoresearch_budget.budget import BudgetLedger


class HarborTrialRunnerTest(unittest.TestCase):
    def _config(
        self, root: Path, paid: bool = False, scaffold: str = "budget-shell"
    ) -> ExperimentConfig:
        value = {
            "budget": {"ledger": "runs/budget.sqlite3", "cap_usd": "2.00"},
            "locations": {"task_root": "tasks", "jobs_root": "runs/trials"},
            "agent": {
                "scaffold": scaffold,
                "version": "2.4.6",
                "max_steps": 5,
                "max_tokens_per_call": 1000,
                "command_timeout_sec": 30,
                "max_observation_chars": 2000,
            },
            "profiles": {
                "small": {
                    "model": "qwen/example",
                    "run_max_cost_usd": "0.25",
                    "service_tier": "flex",
                    "provider": {"allow_fallbacks": False},
                }
            },
            "paid_inference_enabled": paid,
        }
        path = root / "experiment.json"
        path.write_text(json.dumps(value))
        return ExperimentConfig.load(path, project_root=root)

    def test_plan_has_budget_controls_and_no_secret(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"HARBOR_BIN": "/bin/harbor"}):
                command = HarborTrialRunner(self._config(root)).command(
                    TrialSpec("owner__repo-1", "small")
                )
            rendered = " ".join(command)
            self.assertIn("run_max_cost_usd=0.25", rendered)
            self.assertIn("allow_paid=true", rendered)
            self.assertIn('provider={"allow_fallbacks":false}', rendered)
            self.assertNotIn("OPENROUTER", rendered)

    def test_execution_is_locked_by_default(self):
        with TemporaryDirectory() as temp:
            runner = HarborTrialRunner(self._config(Path(temp), paid=False))
            with self.assertRaises(RunnerConfigError):
                runner.run(TrialSpec("owner__repo-1", "small"))

    def test_mini_swe_plan_pins_scaffold_provider_and_limits(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.dict(os.environ, {"HARBOR_BIN": "/bin/harbor"}):
                command = HarborTrialRunner(
                    self._config(root, scaffold="mini-swe-agent")
                ).command(TrialSpec("owner__repo-1", "small"))
            rendered = " ".join(command)
            self.assertIn(
                "--agent autoresearch_budget.harbor_mini_agent:BudgetMiniSweAgent",
                rendered,
            )
            self.assertIn("--model openrouter/qwen/example", rendered)
            self.assertIn("version=2.4.6", rendered)
            self.assertIn("cost_limit=0.25", rendered)
            self.assertIn("max_tokens=1000", rendered)
            self.assertIn('"step_limit":5', rendered)
            self.assertIn('"provider":{"allow_fallbacks":false}', rendered)
            self.assertIn('"service_tier":"flex"', rendered)
            self.assertIn('"OPENROUTER_API_KEY":""', rendered)
            self.assertNotIn("test-key", rendered)

    def test_env_parser_does_not_expand_or_execute_values(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / ".env"
            path.write_text('OPENROUTER_KEY="literal-$VALUE"\n')
            self.assertEqual(
                _read_env_file(path), {"OPENROUTER_KEY": "literal-$VALUE"}
            )

    def test_run_stages_task_with_agent_network_disabled(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "tasks" / "owner__repo-1"
            task.mkdir(parents=True)
            original = (
                'schema_version = "1.3"\n\n'
                '[agent]\n'
                'timeout_sec = 300\n\n'
                '[environment]\n'
                'network_mode = "public"\n'
            )
            (task / "task.toml").write_text(original)
            (task / "instruction.md").write_text("Fix it")
            (root / ".env").write_text("OPENROUTER_API_KEY=test-key\n")
            runner = HarborTrialRunner(self._config(root, paid=True))
            completed = Mock(returncode=0)
            with patch("subprocess.run", return_value=completed) as run:
                self.assertEqual(
                    runner.run(TrialSpec("owner__repo-1", "small")), 0
                )
            command = run.call_args.args[0]
            staged = Path(command[command.index("--path") + 1])
            staged_manifest = (staged / "task.toml").read_text()
            self.assertIn('[agent]\nnetwork_mode = "no-network"\n', staged_manifest)
            self.assertIn('[environment]\nnetwork_mode = "public"\n', staged_manifest)
            self.assertEqual((task / "task.toml").read_text(), original)

    def test_mini_swe_run_allowlists_openrouter_and_reconciles_cost(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            task = root / "tasks" / "owner__repo-1"
            task.mkdir(parents=True)
            original = (
                'schema_version = "1.3"\n\n'
                '[agent]\n'
                'timeout_sec = 300\n\n'
                '[environment]\n'
                'network_mode = "public"\n'
            )
            (task / "task.toml").write_text(original)
            (task / "instruction.md").write_text("Fix it")
            (root / ".env").write_text("OPENROUTER_API_KEY=test-key\n")
            config = self._config(root, paid=True, scaffold="mini-swe-agent")
            runner = HarborTrialRunner(config)

            def completed_run(command, **kwargs):
                del kwargs
                staged = Path(command[command.index("--path") + 1])
                staged_manifest = (staged / "task.toml").read_text()
                self.assertIn('network_mode = "allowlist"', staged_manifest)
                self.assertIn('allowed_hosts = ["openrouter.ai"]', staged_manifest)
                self.assertIn('[environment]\nnetwork_mode = "public"', staged_manifest)
                result = (
                    root
                    / "runs"
                    / "trials"
                    / "small-owner__repo-1-a1"
                    / "result.json"
                )
                result.parent.mkdir(parents=True)
                result.write_text(
                    json.dumps(
                        {
                            "finished_at": "2026-10-03T00:00:00Z",
                            "n_total_trials": 1,
                            "stats": {"cost_usd": 0.12},
                        }
                    )
                )
                return Mock(returncode=0)

            with patch("subprocess.run", side_effect=completed_run):
                self.assertEqual(runner.run(TrialSpec("owner__repo-1", "small")), 0)
            snapshot = BudgetLedger(config.budget_db, config.budget_cap_usd).snapshot()
            self.assertEqual(str(snapshot.settled_usd), "0.120000")
            self.assertEqual(str(snapshot.held_usd), "0.000000")
            self.assertEqual((task / "task.toml").read_text(), original)

    def _poc_config(self, root, *, paid=False):
        config=self._config(root,paid=paid,scaffold="mini-swe-advisor")
        raw=json.loads(config.path.read_text())
        raw["budget"]["enforce"]=False
        raw["agent"]["max_steps"]=80
        raw["profiles"]["frontier"]={"model":"openai/mock-advisor","run_max_cost_usd":"2.00",
            "reasoning_effort":"medium","service_tier":"flex","provider":{"only":["OpenAI"],"allow_fallbacks":False}}
        raw["consultation"]={"policy":"fixed_interval","every_steps":10,"max_calls":100,
            "advisor_profile":"frontier","task_manifest":"sample.json"}
        (root/"sample.json").write_text(json.dumps({"tasks":[{"task_id":"owner__repo-1"}]}))
        config.path.write_text(json.dumps(raw))
        return ExperimentConfig.load(config.path,project_root=root)

    def test_poc_plan_pins_structured_classes_interval_and_run_identity(self):
        with TemporaryDirectory() as temp:
            config=self._poc_config(Path(temp))
            runner=HarborTrialRunner(config)
            command=runner.command(TrialSpec("owner__repo-1","small"))
            value=json.loads(next(arg[7:] for arg in command if arg.startswith("config=")))
            self.assertEqual(value["agent"]["step_limit"],80)
            self.assertEqual(value["agent"]["advice"]["every_steps"],10)
            self.assertEqual(value["agent"]["advice"]["max_calls"],100)
            self.assertEqual(value["agent"]["advice"]["advisor"]["reasoning_effort"],"medium")
            self.assertEqual(value["agent"]["advice"]["run_id"],command[command.index("--job-name")+1])
            self.assertIn("StructuredActorModel",value["model"]["model_class"])
            self.assertIn("PocEnvironment",value["environment"]["environment_class"])
            self.assertIn("cost_limit=0",command)
            with self.assertRaises(RunnerConfigError):
                runner.command(TrialSpec("outside-sample","small"))
            with self.assertRaises(RunnerConfigError):
                runner.run(TrialSpec("owner__repo-1","small"))

    def test_revised_config_carries_fallback_to_harbor_without_changing_actor(self):
        root=Path(__file__).resolve().parents[1]
        config=ExperimentConfig.load(root/'config/experiment.phase1.v2.json',project_root=root)
        original=ExperimentConfig.load(root/'config/experiment.phase1.json',project_root=root)
        command=HarborTrialRunner(config).command(TrialSpec('gotd__td-1759','small'))
        settings=json.loads(next(arg[7:] for arg in command if arg.startswith('config=')))
        advisor=settings['agent']['advice']['advisor']
        self.assertEqual(advisor['model'],'openai/gpt-6.1-sol')
        self.assertEqual(advisor['service_tier'],'flex')
        self.assertEqual(advisor['fallback']['provider']['only'],['azure'])
        self.assertEqual(advisor['fallback']['max_tokens_parameter'],'max_completion_tokens')
        self.assertEqual(settings['agent']['step_limit'],80)
        self.assertEqual(settings['agent']['advice']['every_steps'],10)
        self.assertEqual(config.profiles['small'],original.profiles['small'])
        self.assertNotEqual(config.jobs_root,original.jobs_root)
        self.assertFalse(config.paid_inference_enabled)
        self.assertFalse(config.budget_enforced)

    def test_poc_retains_grading_after_run_without_budget_ledger_or_overwrites(self):
        with TemporaryDirectory() as temp:
            root=Path(temp); config=self._poc_config(root,paid=True)
            task=root/"tasks"/"owner__repo-1"; task.mkdir(parents=True)
            (task/"task.toml").write_text('[agent]\ntimeout_sec=300\n[environment]\nnetwork_mode="public"\n')
            (task/"instruction.md").write_text('Fix the public issue')
            (root/".env").write_text('OPENROUTER_API_KEY=offline-secret\n')
            runner=HarborTrialRunner(config)
            spec=TrialSpec("owner__repo-1","small")
            job=runner._job_name(spec)
            def completed(command,**kwargs):
                self.assertIn('offline-secret',kwargs['env']['OPENROUTER_API_KEY'])
                kwargs['stdout'].write('redact offline-secret\n')
                agent=config.jobs_root/job/'trial'/'agent'; agent.mkdir(parents=True)
                (agent/'mini-swe-agent.trajectory.json').write_text(json.dumps({'info':{'poc_metrics':{'total_cost_usd':'0.10','run_id':job}}}))
                verifier=agent.parent/'verifier'; verifier.mkdir()
                (verifier/'reward.txt').write_text('1.0')
                (verifier/'output.log').write_text('Keep evidence; redact offline-secret')
                (agent.parent/'result.json').write_text(json.dumps({'verifier_result':{'rewards':{'reward':1.0}}}))
                (config.jobs_root/job/'result.json').write_text(json.dumps({'finished_at':'now'}))
                return Mock(returncode=0)
            with patch('subprocess.run',side_effect=completed), patch('autoresearch_budget.runner.BudgetLedger',side_effect=AssertionError('No dollar enforcement')):
                self.assertEqual(runner.run(spec),0)
            records=config.jobs_root/'_poc_records'/job
            metadata=json.loads((records/'metadata.json').read_text())
            self.assertFalse(metadata['configuration']['budget']['enforce'])
            self.assertNotIn('offline-secret',(records/'harbor-console.log').read_text())
            self.assertEqual((records/'post_submission'/'trial'/'verifier'/'reward.txt').read_text(),'1.0')
            self.assertNotIn('offline-secret',(records/'post_submission'/'trial'/'verifier'/'output.log').read_text())
            events=[json.loads(line) for line in (records/'events.jsonl').read_text().splitlines()]
            self.assertTrue(any(e['kind']=='post_submission_verifier' and e['payload']['track']=='post_submission' for e in events))
            self.assertEqual(json.loads((records/'summary.json').read_text())['status'],'completed')
            with patch('subprocess.run',side_effect=AssertionError('Must preserve first run')):
                with self.assertRaises(RunnerConfigError):
                    runner.run(spec)
            self.assertNotEqual(runner._job_name(spec),runner._job_name(TrialSpec("owner__repo-1","small",2)))

    def test_completed_job_cost_rejects_unfinished_result(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            path.write_text(
                json.dumps({"n_total_trials": 1, "stats": {"cost_usd": 0.1}})
            )
            self.assertIsNone(_completed_job_cost(path))

    def test_completed_job_cost_rejects_null_cost(self):
        with TemporaryDirectory() as temp:
            path = Path(temp) / "result.json"
            path.write_text(
                json.dumps(
                    {
                        "finished_at": "2026-10-03T00:00:00Z",
                        "n_total_trials": 1,
                        "stats": {"cost_usd": None},
                    }
                )
            )
            self.assertIsNone(_completed_job_cost(path))

    def test_completed_job_cost_prefers_openrouter_trajectory_billing(self):
        with TemporaryDirectory() as temp:
            job = Path(temp)
            path = job / "result.json"
            path.write_text(json.dumps({
                "finished_at": "2026-10-04T00:00:00Z",
                "n_total_trials": 1,
                "stats": {"cost_usd": 0},
            }))
            agent = job / "trial" / "agent"
            agent.mkdir(parents=True)
            (agent / "mini-swe-agent.trajectory.json").write_text(json.dumps({
                "info": {"model_stats": {"api_calls": 2}},
                "messages": [
                    {"extra": {"response": {"usage": {"cost": 0.01}}}},
                    {"extra": {"response": {"usage": {"cost": 0.02}}}},
                ],
            }))
            self.assertEqual(_completed_job_cost(path), Decimal("0.03"))


if __name__ == "__main__":
    unittest.main()
