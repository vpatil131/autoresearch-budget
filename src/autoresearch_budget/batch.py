"""Bounded advisor experiment scheduling; only the coordinator writes records."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Callable

from .runner import ExperimentConfig, HarborTrialRunner, RunnerConfigError, TrialSpec, _read_env_file
from .store import RunStore
from .advisor import AdvicePolicy

VALID_STOPS = {"Submitted", "LimitsExceeded", "TimeExceeded", "RepeatedFormatError"}
MAX_CONCURRENCY = 6


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def read_lines(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    # A damaged journal is ambiguous: fail closed instead of launching duplicates.
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def append(path: Path, value: dict) -> None:
    with path.open("a") as stream:
        stream.write(json.dumps(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def valid(record: dict) -> bool:
    return record.get("return_code") == 0 and (record.get("metrics") or {}).get("stop_reason") in VALID_STOPS


@contextmanager
def coordinator_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RunnerConfigError("another batch coordinator owns this jobs directory") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


class BatchRunner:
    def __init__(self, config: ExperimentConfig, batch_dir: Path, *, profile="small",
                 concurrency=4, env_file=".env", execute: Callable | None = None):
        self.config = config
        self.batch = batch_dir.resolve()
        self.profile = profile
        self.concurrency = concurrency
        self.env_file = env_file
        self.runner = HarborTrialRunner(config)
        self.execute = execute or self._execute
        if concurrency not in range(1, MAX_CONCURRENCY + 1):
            raise RunnerConfigError(f"batch concurrency must be between 1 and {MAX_CONCURRENCY}")
        if config.agent.get("scaffold") != "mini-swe-advisor":
            raise RunnerConfigError("batch execution requires mini-swe-advisor")
        self.policy = AdvicePolicy.load(config.consultation)
        manifest = config.project_root / config.consultation["task_manifest"]
        self.manifest = read_json(manifest)
        self.tasks = [t["task_id"] for t in self.manifest["tasks"]]
        if not self.tasks or len(set(self.tasks)) != len(self.tasks):
            raise RunnerConfigError("task manifest must contain distinct tasks")
        self.raw_config = config.resolved_config()
        self.fingerprint = hashlib.sha256(json.dumps({"config": self.raw_config, "manifest": self.manifest,
                                                     "profile": profile}, sort_keys=True).encode()).hexdigest()
        for task in self.tasks:
            self.runner.command(TrialSpec(task, profile))
            directory = config.task_root / task
            if not all((directory / filename).is_file() for filename in ("task.toml", "instruction.md")):
                raise RunnerConfigError(f"task package is incomplete: {task}")

    def paths(self, job: str) -> dict:
        return {"job_dir": str(self.config.jobs_root / job),
                "runner_records": str(self.config.jobs_root / "_poc_records" / job)}

    def _claim(self, task: str, attempt: int) -> dict:
        job = self.runner._job_name(TrialSpec(task, self.profile, attempt))
        return {"task_id": task, "profile": self.profile, "attempt": attempt,
                "job_name": job, **self.paths(job)}

    def _collect(self, claim: dict, return_code=None, error=None) -> dict | None:
        summary = read_json(Path(claim["runner_records"]) / "summary.json", {})
        # A terminal runner summary is written after Harbor exits and grading is captured.
        if return_code is None and "return_code" not in summary:
            return None
        trial_paths = list(Path(claim["job_dir"]).glob("*/result.json"))
        result = read_json(trial_paths[0], {}) if len(trial_paths) == 1 else {}
        return {**claim, "return_code": return_code if return_code is not None else summary["return_code"],
                "metrics": summary.get("metrics") or {}, "runner_summary": summary,
                "result": result, "error": error, "finished_at": now()}

    def plan(self) -> dict:
        metadata = read_json(self.batch / "metadata.json", {})
        if metadata.get("fingerprint") and metadata["fingerprint"] != self.fingerprint:
            raise RunnerConfigError("batch configuration, profile or task manifest changed; use a new batch directory")
        if metadata and not metadata.get("fingerprint"):
            # Accept a legacy sequential batch only with its exact frozen settings.
            if (read_json(self.batch / "config.json") != self.raw_config
                    or metadata.get("tasks") != self.tasks):
                raise RunnerConfigError("legacy batch configuration or task list changed")
        records = read_lines(self.batch / "attempts.jsonl")
        claims = read_lines(self.batch / "claims.jsonl")
        for entry in [*records, *claims]:
            expected = self._claim(entry["task_id"], entry["attempt"])
            if entry["task_id"] not in self.tasks or entry["job_name"] != expected["job_name"]:
                raise RunnerConfigError("batch journal does not match the task manifest and condition")
        known = {r["job_name"] for r in records}
        unresolved = []
        for claim in claims:
            if claim["job_name"] not in known:
                recovered = self._collect(claim)
                if recovered is None:
                    unresolved.append(claim)
                else:
                    records.append(recovered)
                    known.add(claim["job_name"])
        completed = {r["task_id"] for r in records if valid(r)}
        failed = {r["task_id"] for r in records if not valid(r)} - completed
        pending = []
        for task in self.tasks:
            if task in completed or task in failed or any(c["task_id"] == task for c in unresolved):
                continue
            attempt = 1
            claim = self._claim(task, attempt)
            while Path(claim["job_dir"]).exists() or Path(claim["runner_records"]).exists():
                # Unindexed existing attempts cannot be safely retried automatically.
                if claim["job_name"] not in known:
                    unresolved.append(claim)
                    break
                attempt += 1
                claim = self._claim(task, attempt)
            else:
                pending.append(claim)
        active_batches = []
        # The old sequential launcher does not take our lock. Its status must block us too.
        for path in (self.config.jobs_root / "_batch").glob("*/status.json"):
            state = read_json(path, {})
            if state.get("state") in {"running", "draining"}:
                active_batches.append(str(path))
        # Our own status can be stale after a crash; claims above must reconcile it.
        own = self.batch / "status.json"
        own_state = read_json(own, {})
        active_batches = [p for p in active_batches if Path(p) != own or own_state.get("current_task")]
        return {"batch_dir": str(self.batch), "concurrency": self.concurrency,
                "total_tasks": len(self.tasks), "completed_tasks": len(completed),
                "policy": self.policy.policy, "max_advice_calls": self.policy.max_calls,
                "every_steps": self.policy.every_steps if self.policy.policy == "fixed_interval" else None,
                "actor_max_steps": self.config.agent["max_steps"],
                "paid_inference_enabled": self.config.paid_inference_enabled,
                "models": {role: {"model": self.config.profiles[name].model,
                                  "provider": self.config.profiles[name].provider,
                                  "service_tier": self.config.profiles[name].service_tier,
                                  "reasoning_effort": self.config.profiles[name].reasoning_effort,
                                  "fallback": self.config.profiles[name].fallback}
                           for role, name in (("actor", self.profile),
                                              ("advisor", self.config.consultation["advisor_profile"]))},
                "pending": pending, "unresolved_attempts": unresolved,
                "failed_tasks": sorted(failed), "active_batches": active_batches,
                "records": records}

    def _execute(self, claim: dict, source: Path) -> dict:
        spec = TrialSpec(claim["task_id"], self.profile, claim["attempt"])
        try:
            frozen = ExperimentConfig.load(self.batch / "config.json", project_root=self.config.project_root)
            code = HarborTrialRunner(frozen).run(spec, env_file=self.env_file, source_root=source)
            return self._collect(claim, code)
        except Exception as error:
            # Never expose arbitrary exception text: it may contain credentials/payloads.
            return self._failure(claim, error)

    @staticmethod
    def _failure(claim: dict, error: Exception) -> dict:
        # A corrupt/missing result must not break recording for other workers.
        return {**claim, "return_code": 1, "metrics": {}, "runner_summary": {},
                "result": {}, "error": type(error).__name__, "finished_at": now()}

    def _snapshot(self, store: RunStore) -> Path:
        source = self.batch / "source" / now().replace(":", "-")
        package = source / "autoresearch_budget"
        package.mkdir(parents=True)
        hashes = {}
        for path in (self.config.project_root / "src/autoresearch_budget").glob("*.py"):
            content = path.read_bytes()
            (package / path.name).write_bytes(content)
            hashes[path.name] = hashlib.sha256(content).hexdigest()
        def git(*args):
            result = subprocess.run(["git", *args], cwd=self.config.project_root,
                                    capture_output=True, text=True, check=False)
            return result.stdout if result.returncode == 0 else None
        diff = git("diff", "--", "src", "tests", "docs", "README.md")
        if diff:
            store.write_artifact("source-diff", {"diff": diff})
        store.append_event("batch_dispatch", {"source": str(source), "sha256": hashes,
                           "git_commit": git("rev-parse", "HEAD"), "concurrency": self.concurrency})
        return source

    def _status(self, state: str, records: list, active: dict, pending: list) -> None:
        RunStore._write_json_atomic(self.batch / "status.json", {
            "state": state, "updated_at": now(), "concurrency": self.concurrency,
            "total_tasks": len(self.tasks), "completed_tasks": len({r["task_id"] for r in records if valid(r)}),
            "current_task": None, "active_tasks": list(active.values()),
            "queued_tasks": [c["task_id"] for c in pending], "tasks": records})

    def run(self) -> int:
        if not self.config.paid_inference_enabled:
            raise RunnerConfigError("paid inference is disabled in the experiment config")
        with coordinator_lock(self.config.jobs_root / "_batch/.coordinator.lock"):
            plan = self.plan()
            if plan["active_batches"] or plan["unresolved_attempts"] or plan["failed_tasks"]:
                raise RunnerConfigError("batch requires reconciliation: active coordinator, unresolved attempt or integration failure; inspect batch plan")
            environment = {**os.environ, **_read_env_file(self.config.project_root / self.env_file)}
            if not (environment.get("OPENROUTER_API_KEY") or environment.get("OPENROUTER_KEY")):
                raise RunnerConfigError("OPENROUTER_API_KEY or OPENROUTER_KEY is missing from the environment")
            secrets = tuple(value for key, value in environment.items()
                            if any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "OPENROUTER_KEY")))
            self.batch.mkdir(parents=True, exist_ok=True)
            store = RunStore(self.batch, secrets=secrets)
            metadata_path = self.batch / "metadata.json"
            if not metadata_path.exists():
                store.write_metadata({"started_at": now(), "fingerprint": self.fingerprint,
                                      "config": self.raw_config, "tasks": self.tasks,
                                      "profile": self.profile, "execution": "parallel"})
                RunStore._write_json_atomic(self.batch / "config.json", self.raw_config)
                RunStore._write_json_atomic(self.batch / "manifest.json", self.manifest)
            records = store.redact(plan["records"])
            existing = {r["job_name"] for r in read_lines(self.batch / "attempts.jsonl")}
            for record in records:
                if record["job_name"] not in existing:
                    append(self.batch / "attempts.jsonl", store.redact(record))
                    store.append_event("attempt_recovered", {"job_name": record["job_name"]})
            pending = list(plan["pending"])
            active = {}
            if not pending:
                self._status("complete", records, active, pending)
                return 0
            source = self._snapshot(store)
            stopped = False
            drain = False
            with ThreadPoolExecutor(max_workers=self.concurrency) as workers:
                while pending or active:
                    drain = drain or (self.batch / "drain.request").exists()
                    while pending and len(active) < self.concurrency and not stopped and not drain:
                        claim = {**pending.pop(0), "started_at": now(), "source_root": str(source)}
                        # Persist the claim before any paid dispatch. Only this thread writes journals.
                        append(self.batch / "claims.jsonl", claim)
                        store.append_event("task_dispatched", claim)
                        future = workers.submit(self.execute, claim, source)
                        active[future] = claim
                    self._status("needs_attention" if stopped else "draining" if drain else "running", records, active, pending)
                    if not active:
                        break
                    finished, _ = wait(active, timeout=1, return_when=FIRST_COMPLETED)
                    for future in finished:
                        claim = active.pop(future)
                        try:
                            record = future.result()
                        except Exception as error:
                            record = self._failure(claim, error)
                        record = store.redact(record)
                        append(self.batch / "attempts.jsonl", record)
                        records.append(record)
                        store.append_event("task_finished", {"job_name": claim["job_name"],
                                           "valid_attempt": valid(record), "return_code": record["return_code"]})
                        stopped = stopped or not valid(record)
            state = "needs_attention" if stopped else "drained" if pending else "complete"
            self._status(state, records, active, pending)
            return 2 if stopped else 0
