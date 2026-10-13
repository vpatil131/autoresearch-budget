"""Contracts required by an installed harness in a separate workspace."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from autoresearch_budget.runner import ExperimentConfig, HarborTrialRunner, TrialSpec
from autoresearch_budget.batch import BatchRunner
from autoresearch_budget.store import RunStore
from autoresearch_budget.docker_runtime import check_runtime


class InstallationTest(unittest.TestCase):
    def test_config_pins_workspace_independently_of_invocation_directory(self):
        with TemporaryDirectory(prefix="workspace with spaces ") as directory:
            root = Path(directory).resolve()
            (root / "config").mkdir()
            raw = json.loads(Path("config/experiment.example.json").read_text())
            raw["workspace"] = ".."
            path = root / "config/experiment.json"
            path.write_text(json.dumps(raw))
            with patch("pathlib.Path.cwd", return_value=Path("/tmp")):
                config = ExperimentConfig.load(path)
            self.assertEqual(config.project_root, root)
            self.assertEqual(config.task_root, root / raw["locations"]["task_root"])
            command = HarborTrialRunner(config).command(TrialSpec("owner__repo-1", "small"))
            self.assertIn("autoresearch_budget.docker_runtime", command)
            self.assertNotIn(str(root / "scripts/with_anonymous_docker.sh"), command)

    def test_snapshot_uses_runtime_sources_when_workspace_has_no_source(self):
        with TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            runner = object.__new__(BatchRunner)
            runner.batch = root / "batch"
            runner.config = type("Config", (), {"project_root": root})()
            runner.concurrency = 1
            source = runner._snapshot(RunStore(runner.batch))
            from autoresearch_budget import runner as runtime
            self.assertEqual((source / "autoresearch_budget/runner.py").read_bytes(),
                             Path(runtime.__file__).read_bytes())
            event = json.loads((runner.batch / "events.jsonl").read_text().splitlines()[-1])
            self.assertTrue(event["payload"]["sha256"])

    def test_runtime_rejects_incompatible_versions(self):
        with patch("autoresearch_budget.docker_runtime.version", return_value="999"):
            with self.assertRaisesRegex(ValueError, "unsupported"):
                check_runtime()


if __name__ == "__main__":
    unittest.main()
