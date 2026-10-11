"""Offline process-lifecycle checks for foreground command execution."""
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

HAS_MINI = importlib.util.find_spec("minisweagent") is not None
if HAS_MINI:
    from autoresearch_budget.poc_environment import PocEnvironment
    from autoresearch_budget.store import RunStore


@unittest.skipUnless(HAS_MINI, "Install mini-SWE-agent 2.4.6 for runtime tests")
class PocEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.env = PocEnvironment(cwd=str(self.root), timeout=600,
                                  env={"OPENROUTER_API_KEY": "", "MSWEA_API_KEY": ""})
        self.store = RunStore(self.root / "records", run_id="command-test")
        self.env.bind(self.store, lambda: 3)

    def command(self, code):
        return f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}"

    def events(self):
        return [json.loads(line)["payload"] for line in self.store.events_path.read_text().splitlines()]

    def test_foreground_completion_returns_without_waiting_for_deadline(self):
        started = time.monotonic()
        output = self.env.execute({"command": self.command(
            "import sys,time; time.sleep(.1); print('x'*100000); print('stderr',file=sys.stderr); sys.exit(7)"),
            "observation_id": "observation-3", "patch_id": "patch-2"})
        self.assertLess(time.monotonic() - started, 3)
        self.assertEqual(output["returncode"], 7)
        self.assertIn("x" * 100000, output["output"])
        self.assertIn("stderr", output["output"])
        self.assertFalse(output["timed_out"])
        event = self.events()[0]
        self.assertEqual((event["patch_id"], event["observation_id"]), ("patch-2", "observation-3"))
        self.assertEqual(event["timeout_seconds"], 600)

    def test_remaining_task_time_limits_command_and_preserves_partial_output(self):
        self.env.set_time_remaining(lambda: .15)
        output = self.env.execute({"command": self.command(
            "import time; print('started',flush=True); time.sleep(10)")})
        self.assertTrue(output["timed_out"])
        self.assertEqual(output["returncode"], -1)
        self.assertIn("started", output["output"])
        self.assertEqual(output["extra"]["exception_type"], "TimeoutExpired")
        self.assertLess(output["duration_seconds"], 3)
        self.assertEqual(self.events()[0]["timeout_seconds"], .15)

    def test_exhausted_task_deadline_does_not_launch_command(self):
        self.env.set_time_remaining(lambda: -1)
        output = self.env.execute({"command": "touch should-not-exist"})
        self.assertTrue(output["timed_out"])
        self.assertFalse((self.root / "should-not-exist").exists())

    def test_credentials_remain_masked_in_child_shell(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "host-secret", "MSWEA_API_KEY": "host-secret"}):
            output = self.env.execute({"command": self.command(
                "import os; assert not os.environ['OPENROUTER_API_KEY']; assert not os.environ['MSWEA_API_KEY']")})
        self.assertEqual(output["returncode"], 0)

    def child_command(self):
        child = "from pathlib import Path; import time; time.sleep(.6); Path('child-survived').touch()"
        return self.command(f"import subprocess,sys,time; from pathlib import Path; "
                            f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
                            "Path('ready').touch(); print('started',flush=True); time.sleep(10)")

    @unittest.skipUnless(os.name == "posix", "Process group cleanup requires POSIX")
    def test_timeout_kills_descendants(self):
        output = self.env.execute({"command": self.child_command()}, timeout=.25)
        self.assertTrue(output["timed_out"])
        self.assertTrue((self.root / "ready").exists())
        time.sleep(.7)
        self.assertFalse((self.root / "child-survived").exists())

    @unittest.skipUnless(os.name == "posix", "Process group cleanup requires POSIX")
    def test_interruption_kills_descendants_and_records_output(self):
        wait = subprocess.Popen.wait

        def interrupt(process, *args, **kwargs):
            if kwargs.get("timeout") is not None:
                deadline = time.monotonic() + 2
                while not (self.root / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue((self.root / "ready").exists())
                raise KeyboardInterrupt()
            return wait(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", interrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.env.execute({"command": self.child_command()})
        self.assertEqual(self.events()[0]["result"]["extra"]["exception_type"], "KeyboardInterrupt")
        time.sleep(.7)
        self.assertFalse((self.root / "child-survived").exists())
