"""Repository execution with full output capture and explicit typed submission."""
from __future__ import annotations

import time
from minisweagent.environments.local import LocalEnvironment


class PocEnvironment(LocalEnvironment):
    def bind(self, store, actor_call):
        self.store = store
        self.actor_call = actor_call

    def _check_finished(self, output):
        # Only the controller's validated submit tool can terminate the attempt.
        pass

    def execute(self, action, *args, **kwargs):
        started = time.monotonic()
        output = super().execute(action, *args, **kwargs)
        if hasattr(self, "store"):
            self.store.append_event("command", {
                "actor_call": self.actor_call(), "source": action.get("source", "actor"),
                "command": action.get("command"), "result": output,
                "duration_seconds": time.monotonic() - started,
                "observation_id": action.get("observation_id"),
            })
        return output
