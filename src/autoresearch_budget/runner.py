"""Build and execute reproducible, single-task Harbor trials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any


class RunnerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    run_max_cost_usd: Decimal
    reasoning_effort: str | None
    provider: dict[str, Any] | None


@dataclass(frozen=True)
class ExperimentConfig:
    path: Path
    project_root: Path
    budget_db: Path
    budget_cap_usd: Decimal
    task_root: Path
    jobs_root: Path
    agent: dict[str, Any]
    profiles: dict[str, ModelProfile]
    paid_inference_enabled: bool

    @classmethod
    def load(cls, path: str | Path, *, project_root: str | Path) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        root = Path(project_root).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        budget = raw["budget"]
        locations = raw.get("locations", {})
        profiles = {
            name: ModelProfile(
                name=name,
                model=value["model"],
                run_max_cost_usd=Decimal(str(value["run_max_cost_usd"])),
                reasoning_effort=value.get("reasoning_effort"),
                provider=value.get("provider"),
            )
            for name, value in raw["profiles"].items()
        }
        return cls(
            path=config_path,
            project_root=root,
            budget_db=_under_root(root, budget["ledger"]),
            budget_cap_usd=Decimal(str(budget["cap_usd"])),
            task_root=_under_root(
                root,
                locations.get(
                    "task_root", "data/harbor/swe-rebench-07-2026"
                ),
            ),
            jobs_root=_under_root(root, locations.get("jobs_root", "runs/trials")),
            agent=dict(raw["agent"]),
            profiles=profiles,
            paid_inference_enabled=bool(raw.get("paid_inference_enabled", False)),
        )


@dataclass(frozen=True)
class TrialSpec:
    task_id: str
    profile: str
    attempt: int = 1


class HarborTrialRunner:
    AGENT_IMPORT = "autoresearch_budget.harbor_agent:BudgetShellAgent"

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def command(
        self, spec: TrialSpec, *, task_dir: Path | None = None
    ) -> list[str]:
        if spec.profile not in self.config.profiles:
            raise RunnerConfigError(f"unknown model profile {spec.profile!r}")
        if spec.attempt < 1:
            raise RunnerConfigError("attempt must be at least 1")
        if "/" in spec.task_id or spec.task_id in (".", ".."):
            raise RunnerConfigError("task_id must be one task directory name")
        profile = self.config.profiles[spec.profile]
        task_dir = task_dir or self.config.task_root / spec.task_id
        job_name = f"{spec.profile}-{spec.task_id}-a{spec.attempt}"
        command = [
            *self._harbor_prefix(),
            "run",
            "--path",
            str(task_dir),
            "--agent",
            self.AGENT_IMPORT,
            "--model",
            profile.model,
            "--ak",
            f"budget_db={self.config.budget_db}",
            "--ak",
            f"budget_cap_usd={self.config.budget_cap_usd}",
            "--ak",
            f"run_max_cost_usd={profile.run_max_cost_usd}",
            "--ak",
            f"max_steps={int(self.config.agent['max_steps'])}",
            "--ak",
            f"max_tokens_per_call={int(self.config.agent['max_tokens_per_call'])}",
            "--ak",
            f"command_timeout_sec={int(self.config.agent['command_timeout_sec'])}",
            "--ak",
            f"max_observation_chars={int(self.config.agent['max_observation_chars'])}",
            "--ak",
            "allow_paid=true",
        ]
        if profile.reasoning_effort:
            command.extend(["--ak", f"reasoning_effort={profile.reasoning_effort}"])
        if profile.provider:
            command.extend(
                [
                    "--ak",
                    "provider=" + json.dumps(profile.provider, separators=(",", ":")),
                ]
            )
        command.extend(
            [
                "--n-concurrent",
                "1",
                "--jobs-dir",
                str(self.config.jobs_root),
                "--job-name",
                job_name,
                "--yes",
            ]
        )
        return command

    def run(self, spec: TrialSpec, *, env_file: str | Path = ".env") -> int:
        if not self.config.paid_inference_enabled:
            raise RunnerConfigError(
                "paid inference is disabled in the experiment config"
            )
        task_dir = self.config.task_root / spec.task_id
        if not task_dir.is_dir():
            raise RunnerConfigError(f"task directory does not exist: {task_dir}")
        prepared_task_dir = self._prepare_isolated_task(spec, task_dir)
        environment = dict(os.environ)
        environment.update(_read_env_file(self.config.project_root / env_file))
        api_key = environment.get("OPENROUTER_API_KEY") or environment.get(
            "OPENROUTER_KEY"
        )
        if not api_key:
            raise RunnerConfigError(
                "OPENROUTER_API_KEY or OPENROUTER_KEY is missing from the environment"
            )
        environment["OPENROUTER_API_KEY"] = api_key
        existing_pythonpath = environment.get("PYTHONPATH")
        source_path = str(self.config.project_root / "src")
        environment["PYTHONPATH"] = (
            source_path + os.pathsep + existing_pythonpath
            if existing_pythonpath
            else source_path
        )
        self.config.jobs_root.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(
            self.command(spec, task_dir=prepared_task_dir),
            cwd=self.config.project_root,
            env=environment,
            check=False,
        )
        return completed.returncode

    def _prepare_isolated_task(self, spec: TrialSpec, source: Path) -> Path:
        """Stage a task whose agent phase cannot reach the public internet."""
        destination = (
            self.config.jobs_root
            / "_prepared_tasks"
            / f"{spec.profile}-{spec.task_id}-a{spec.attempt}"
        )
        if destination.exists():
            shutil.rmtree(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        manifest = destination / "task.toml"
        if not manifest.is_file():
            raise RunnerConfigError(f"task manifest does not exist: {manifest}")
        manifest.write_text(
            _force_agent_no_network(manifest.read_text(encoding="utf-8")),
            encoding="utf-8",
        )
        return destination

    def _harbor_prefix(self) -> list[str]:
        wrapper = str(self.config.project_root / "scripts/with_anonymous_docker.sh")
        explicit = os.environ.get("HARBOR_BIN")
        if explicit:
            return [wrapper, explicit]
        installed = shutil.which("harbor")
        if installed:
            return [wrapper, installed]
        return [wrapper, "uvx", "harbor==0.23.0"]


def _under_root(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _read_env_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise RunnerConfigError(f"invalid .env line {number}")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        values[key] = value
    return values


def _force_agent_no_network(source: str) -> str:
    """Set Harbor's agent-phase network policy without rewriting unrelated TOML."""
    lines = source.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "[agent]"),
        None,
    )
    if start is None:
        raise RunnerConfigError("task.toml has no [agent] section")
    end = next(
        (
            index
            for index in range(start + 1, len(lines))
            if lines[index].lstrip().startswith("[")
        ),
        len(lines),
    )
    for index in range(start + 1, end):
        if lines[index].split("=", 1)[0].strip() == "network_mode":
            newline = "\n" if lines[index].endswith("\n") else ""
            lines[index] = f'network_mode = "no-network"{newline}'
            return "".join(lines)
    lines.insert(start + 1, 'network_mode = "no-network"\n')
    return "".join(lines)
