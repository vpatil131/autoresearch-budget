"""Safe initialization and local execution-environment checks."""
from __future__ import annotations

from importlib.resources import files
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

from .advisor import AdvicePolicy
from .docker_runtime import check_runtime
from .runner import ExperimentConfig, HarborTrialRunner, TrialSpec, _read_env_file
from .trial_pool import select_tasks

PRESETS = ("standalone-small", "standalone-frontier", "scheduled", "requested",
           "plan-review", "recovery-review", "hybrid", "control")
POLICY_NAMES = {"scheduled": "fixed_interval", "requested": "advice_requested",
                "plan-review": "plan_review", "recovery-review": "recovery_review",
                "hybrid": "hybrid", "control": "phase3_control"}


def initialize(workspace: Path, preset: str) -> dict:
    workspace = workspace.expanduser().resolve()
    if preset not in PRESETS:
        raise ValueError(f"unknown preset: {preset}")
    resource = files("autoresearch_budget").joinpath("resources")
    standalone = preset.startswith("standalone-")
    raw = json.loads(resource.joinpath("experiment.example.json" if standalone else "experiment.phase3.json").read_text())
    raw["workspace"] = "."
    raw["locations"]["jobs_root"] = f"runs/{preset}"
    raw["paid_inference_enabled"] = False
    profile = "frontier" if preset == "standalone-frontier" else "small"
    if not standalone:
        raw["budget"].pop("cap_usd", None)
        for value in raw["profiles"].values():
            value.pop("run_max_cost_usd", None)
        raw["consultation"]["policy"] = POLICY_NAMES[preset]
        raw["consultation"]["task_manifest"] = "tasks.json"
        raw["consultation"]["max_calls"] = 0 if preset == "control" else 7
        if preset == "scheduled":
            raw["consultation"]["every_steps"] = 10
    destinations = {"experiment.local.json": json.dumps(raw, indent=2) + "\n",
                    "tasks.json": resource.joinpath("sample-24-v2.json").read_text(),
                    ".env.example": "# Copy to .env and set a key for explicit paid runs.\nOPENROUTER_API_KEY=\n",
                    ".gitignore": ".env\n*.local.json\nruns/\ndata/\n"}
    # Check all conflicts before writing anything. Do not touch existing .env.
    conflicts = [name for name in destinations if (workspace / name).exists()]
    if conflicts:
        raise ValueError("initialization would overwrite: " + ", ".join(conflicts))
    workspace.mkdir(parents=True, exist_ok=True)
    for name, content in destinations.items():
        with (workspace / name).open("x", encoding="utf-8") as stream:
            stream.write(content)
    return {"workspace": str(workspace), "preset": preset, "profile": profile,
            "config": str(workspace / "experiment.local.json"),
            "paid_inference_enabled": False,
            "budget_enforced": standalone,
            "next": shlex.join(["autoresearch-budget", "doctor", "--config", str(workspace / "experiment.local.json")])}


def doctor(config_path: Path | None, *, offline: bool = False, env_file: str = ".env") -> dict:
    checks = []
    def add(name, ok, detail):
        checks.append({"check": name, "ok": bool(ok), "detail": detail})
    add("python", sys.version_info >= (3, 12), f"{sys.version_info.major}.{sys.version_info.minor}")
    add("platform", sys.platform in {"darwin", "linux"}, sys.platform)
    add("git", shutil.which("git") is not None, "required by the advisory demo and coding tasks")
    try:
        check_runtime()
        add("runtime", True, "Harbor 0.23.0 and mini-SWE-agent 2.4.6")
    except ValueError as error:
        add("runtime", False, str(error))
    config = None
    if config_path:
        try:
            config = ExperimentConfig.load(config_path)
            manifest = config.consultation.get("task_manifest")
            task_ids = select_tasks(None, config.project_root / manifest) if manifest else []
            for task_id in task_ids:
                HarborTrialRunner(config).command(TrialSpec(task_id, next(iter(config.profiles))))
            add("configuration", True, str(config.path))
        except (OSError, ValueError, KeyError, TypeError) as error:
            add("configuration", False, str(error))
            config = None
    if not offline:
        for name, command in (("docker", ["docker", "info", "--format", "{{.ServerVersion}}"]),
                              ("compose", ["docker", "compose", "version", "--short"])):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=20, check=False)
                add(name, result.returncode == 0, result.stdout.strip() if result.returncode == 0 else "unavailable")
            except (OSError, subprocess.SubprocessError):
                add(name, False, "unavailable")
        if config:
            environment = os.environ | _read_env_file(config.project_root / env_file)
            add("credentials", bool(environment.get("OPENROUTER_API_KEY") or environment.get("OPENROUTER_KEY")),
                "presence only; no provider request made")
            manifest = config.consultation.get("task_manifest")
            if manifest:
                from .dataset import inspect_tasks
                try:
                    inspection = inspect_tasks(config.task_root, config.project_root / manifest)
                    add("task_packages", True, f"{len(inspection['tasks'])} selected tasks validated")
                except (OSError, ValueError, KeyError, TypeError) as error:
                    add("task_packages", False, str(error))
            else:
                add("task_root", config.task_root.is_dir(), str(config.task_root))
    return {"ok": all(c["ok"] for c in checks), "offline": offline, "checks": checks,
            "paid_inference_enabled": config.paid_inference_enabled if config else False,
            "budget_enforced": config.budget_enforced if config else None,
            "accounting_mode": "reserved" if config and config.budget_enforced else "observational; dollars are not capped" if config else None}
