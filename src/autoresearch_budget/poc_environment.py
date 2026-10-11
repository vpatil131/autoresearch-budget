"""Repository execution with full output capture and explicit typed submission."""
from __future__ import annotations

import os
import signal
import subprocess
import tempfile
import time
from minisweagent.environments.local import LocalEnvironment


class PocEnvironment(LocalEnvironment):
    def bind(self, store, actor_call):
        self.store = store
        self.actor_call = actor_call

    def set_time_remaining(self, callback):
        self.time_remaining = callback

    def _check_finished(self, output):
        # Only the controller's validated submit tool can terminate the attempt.
        pass

    @staticmethod
    def _stop(process):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        process.wait()

    @staticmethod
    def _failure(error, text):
        return {"output": text, "returncode": -1,
                "exception_info": f"An error occurred while executing the command: {error}",
                "extra": {"exception_type": type(error).__name__, "exception": str(error)}}

    def execute(self, action, cwd="", *, timeout=None):
        started = time.monotonic()
        command = action.get("command", "")
        effective_timeout = self.config.timeout if timeout is None else timeout
        if callback := getattr(self, "time_remaining", None):
            remaining = callback()
            if remaining is not None:
                effective_timeout = min(effective_timeout, max(0, remaining))
        output = None
        try:
            if effective_timeout <= 0:
                raise subprocess.TimeoutExpired(command, effective_timeout)
            # Capture continuously on disk, including output larger than a pipe
            # buffer. Waiting is controller work and requires no actor polling.
            with tempfile.TemporaryFile() as capture:
                process = subprocess.Popen(command, shell=True,
                    cwd=cwd or self.config.cwd or os.getcwd(), env=os.environ | self.config.env,
                    stdout=capture, stderr=subprocess.STDOUT, start_new_session=os.name == "posix")
                try:
                    process.wait(timeout=effective_timeout)
                except BaseException as error:
                    self._stop(process)
                    capture.seek(0)
                    text = capture.read().decode("utf-8", errors="replace")
                    if isinstance(error, subprocess.TimeoutExpired):
                        error.output = text
                    output = self._failure(error, text)
                    raise
                capture.seek(0)
                output = {"output": capture.read().decode("utf-8", errors="replace"),
                          "returncode": process.returncode, "exception_info": ""}
        except Exception as error:
            output = self._failure(error, getattr(error, "output", None) or "")
        finally:
            if output is not None:
                output["timed_out"] = output.get("extra", {}).get("exception_type") == "TimeoutExpired"
                output["duration_seconds"] = time.monotonic() - started
                if hasattr(self, "store"):
                    self.store.append_event("command", {
                        "actor_call": self.actor_call(), "source": action.get("source", "actor"),
                        "command": command, "result": output,
                        "duration_seconds": output["duration_seconds"],
                        "observation_id": action.get("observation_id"),
                        "patch_id": action.get("patch_id"), "timed_out": output["timed_out"],
                        "timeout_seconds": effective_timeout,
                    })
        self._check_finished(output)
        return output
