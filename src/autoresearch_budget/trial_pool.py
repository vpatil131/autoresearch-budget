"""Bounded parallel execution of independent Harbor trials."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
from typing import Callable

from .budget import BudgetLedger
from .runner import HarborTrialRunner, RunnerConfigError, TrialSpec

MAX_WORKERS = 8


def select_tasks(task_ids: list[str] | None, manifest: Path | None) -> list[str]:
    if manifest is not None:
        data = json.loads(manifest.read_text())
        task_ids = [task["task_id"] for task in data["tasks"]]
    if not task_ids or any(not isinstance(task, str) or not task.strip() for task in task_ids):
        raise RunnerConfigError("select at least one nonempty task ID")
    if any("/" in task or "\\" in task or task in {".", ".."} or task != task.strip()
           or any(ord(char) < 32 for char in task) for task in task_ids):
        raise RunnerConfigError("task IDs must be plain directory names")
    if len(set(task_ids)) != len(task_ids):
        raise RunnerConfigError("task IDs must be distinct")
    return task_ids


class TrialPool:
    def __init__(self, runner: HarborTrialRunner, specs: list[TrialSpec], *, workers: int = 1,
                 env_file: str = ".env"):
        if workers not in range(1, MAX_WORKERS + 1):
            raise RunnerConfigError(f"trial workers must be between 1 and {MAX_WORKERS}")
        if not specs or len({spec.task_id for spec in specs}) != len(specs):
            raise RunnerConfigError("select distinct tasks for trial execution")
        self.runner, self.specs = runner, specs
        self.workers, self.env_file = workers, env_file
        # Validate every command before any paid work can start.
        self.commands = [runner.command(spec) for spec in specs]

    def plan(self) -> dict:
        config = self.runner.config
        return {"budget_enforced": config.budget_enforced,
                "accounting_mode": "reserved" if config.budget_enforced else "observational; dollars are not capped",
                "paid_inference_enabled": config.paid_inference_enabled,
                "workers": self.workers, "effective_workers": min(self.workers, len(self.specs)),
                "trials": [{"task_id": spec.task_id, "profile": spec.profile,
                            "attempt": spec.attempt, "command": command}
                           for spec, command in zip(self.specs, self.commands)]}

    def run(self, report: Callable[[dict], None] | None = None) -> int:
        config = self.runner.config
        if not config.paid_inference_enabled:
            raise RunnerConfigError("paid inference is disabled in the experiment config")
        for spec in self.specs:
            directory = config.task_root / spec.task_id
            if not all((directory / filename).is_file() for filename in ("task.toml", "instruction.md")):
                raise RunnerConfigError(f"task package is incomplete: {spec.task_id}")
            job = self.runner._job_name(spec)
            if (config.jobs_root / job).exists() or (config.jobs_root / "_poc_records" / job).exists() or (config.jobs_root / "_run_records" / job).exists():
                raise RunnerConfigError(f"trial attempt already exists: {job}; use a new attempt or jobs directory")

        # Initialize a new shared ledger once before concurrent reservations.
        if config.budget_enforced:
            BudgetLedger(config.budget_db, config.budget_cap_usd)

        remaining = iter(self.specs)
        failed = False
        completed = 0
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            active = {}

            def dispatch() -> bool:
                spec = next(remaining, None)
                if spec is None:
                    return False
                future = executor.submit(self.runner.run, spec, env_file=self.env_file)
                active[future] = spec
                return True

            for _ in range(min(self.workers, len(self.specs))):
                dispatch()
            while active:
                done, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in done:
                    spec = active.pop(future)
                    try:
                        code = future.result()
                        error = None
                    except Exception as exception:
                        code, error = 1, type(exception).__name__
                    completed += 1
                    failed = failed or code != 0
                    if report:
                        report({"task_id": spec.task_id, "profile": spec.profile,
                                "attempt": spec.attempt, "return_code": code, "error": error,
                                "completed": completed, "total": len(self.specs)})
                # Stop new dispatch after a runner failure; active attempts finish.
                if not failed:
                    while len(active) < self.workers and dispatch():
                        pass
        return 1 if failed else 0
