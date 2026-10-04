import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from autoresearch_budget.runner import (
    ExperimentConfig,
    HarborTrialRunner,
    RunnerConfigError,
    TrialSpec,
    _read_env_file,
)


class HarborTrialRunnerTest(unittest.TestCase):
    def _config(self, root: Path, paid: bool = False) -> ExperimentConfig:
        value = {
            "budget": {"ledger": "runs/budget.sqlite3", "cap_usd": "2.00"},
            "locations": {"task_root": "tasks", "jobs_root": "runs/trials"},
            "agent": {
                "max_steps": 5,
                "max_tokens_per_call": 1000,
                "command_timeout_sec": 30,
                "max_observation_chars": 2000,
            },
            "profiles": {
                "small": {
                    "model": "qwen/example",
                    "run_max_cost_usd": "0.25",
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


if __name__ == "__main__":
    unittest.main()
