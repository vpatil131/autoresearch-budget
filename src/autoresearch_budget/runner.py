"""Build and execute reproducible, single-task Harbor trials."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import sys
from dataclasses import dataclass, field, replace
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from .budget import BudgetLedger
from .advisor import AdvicePolicy, validate_advisor_routes
from .store import RunStore
from .provenance import trial_provenance, package_hash
from .processes import run_command


class RunnerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    run_max_cost_usd: Decimal
    reasoning_effort: str | None
    service_tier: str | None
    provider: dict[str, Any] | None
    fallback: dict[str, Any] | None = None


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
    budget_enforced: bool = True
    consultation: dict[str, Any] = field(default_factory=dict)
    source_config: dict[str, Any] = field(default_factory=dict, repr=False)

    def resolved_config(self) -> dict:
        raw = deepcopy(self.source_config or json.loads(self.path.read_text()))
        if self.consultation or "consultation" in raw:
            raw["consultation"] = deepcopy(self.consultation)
        return raw

    def with_consultation_overrides(self, *, policy=None, max_calls=None, every_steps=None) -> "ExperimentConfig":
        if policy is None and max_calls is None and every_steps is None:
            return self
        if self.agent.get("scaffold") != "mini-swe-advisor":
            raise RunnerConfigError("advice overrides require mini-swe-advisor")
        consultation = dict(self.consultation)
        if policy is not None:
            consultation["policy"] = policy
        if max_calls is not None:
            consultation["max_calls"] = max_calls
        if consultation.get("policy", "fixed_interval") != "fixed_interval":
            if every_steps is not None:
                raise RunnerConfigError("--advice-every-steps applies only to fixed_interval")
            consultation.pop("every_steps", None)
        elif every_steps is not None:
            consultation["every_steps"] = every_steps
        AdvicePolicy.load(consultation)
        return replace(self, consultation=consultation)

    @classmethod
    def load(cls, path: str | Path, *, project_root: str | Path | None = None) -> "ExperimentConfig":
        config_path = Path(path).resolve()
        raw = json.loads(config_path.read_text(encoding="utf-8"))
        _validate_config(raw)
        # Explicit API root wins. New configs pin their workspace relative to the
        # config file; legacy configs retain their invocation-directory behavior.
        root = (Path(project_root).resolve() if project_root is not None else
                _under_root(config_path.parent, raw["workspace"]) if "workspace" in raw else Path.cwd().resolve())
        budget = raw["budget"]
        locations = raw.get("locations", {})
        profiles = {
            name: ModelProfile(
                name=name,
                model=value["model"],
                run_max_cost_usd=Decimal(str(value.get("run_max_cost_usd", 0))),
                reasoning_effort=value.get("reasoning_effort"),
                service_tier=value.get("service_tier"),
                provider=value.get("provider"),
                fallback=value.get("fallback"),
            )
            for name, value in raw["profiles"].items()
        }
        return cls(
            path=config_path,
            project_root=root,
            budget_db=_under_root(root, budget["ledger"]),
            budget_cap_usd=Decimal(str(budget.get("cap_usd", 0))),
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
            budget_enforced=bool(budget.get("enforce", True)),
            consultation=dict(raw.get("consultation", {})),
            source_config=raw,
        )


@dataclass(frozen=True)
class TrialSpec:
    task_id: str
    profile: str
    attempt: int = 1


class HarborTrialRunner:
    AGENT_IMPORT = "autoresearch_budget.harbor_agent:BudgetShellAgent"
    MINI_SWE_AGENT = "mini-swe-agent"
    MINI_SWE_AGENT_IMPORT = (
        "autoresearch_budget.harbor_mini_agent:BudgetMiniSweAgent"
    )
    ADVISORY_SCAFFOLD = "mini-swe-advisor"
    ADVISORY_AGENT_IMPORT = "autoresearch_budget.harbor_mini_agent:AdvisoryMiniSweAgent"
    SUPPORTED_SCAFFOLDS = {"budget-shell", MINI_SWE_AGENT, ADVISORY_SCAFFOLD}

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def command(
        self, spec: TrialSpec, *, task_dir: Path | None = None
    ) -> list[str]:
        if spec.profile not in self.config.profiles:
            raise RunnerConfigError(f"unknown model profile {spec.profile!r}")
        if type(spec.attempt) is not int or spec.attempt < 1:
            raise RunnerConfigError("attempt must be at least 1")
        if not _directory_name(spec.task_id):
            raise RunnerConfigError("task_id must be one task directory name")
        profile = self.config.profiles[spec.profile]
        task_dir = task_dir or self.config.task_root / spec.task_id
        job_name = self._job_name(spec)
        scaffold = self._scaffold()
        if scaffold == self.ADVISORY_SCAFFOLD:
            self._validate_poc(spec)
        elif not self.config.budget_enforced:
            raise RunnerConfigError("budget.enforce=false is supported only by mini-swe-advisor")
        command = [
            *self._harbor_prefix(),
            "run",
            "--path",
            str(task_dir),
            "--agent",
            (
                self.ADVISORY_AGENT_IMPORT
                if scaffold == self.ADVISORY_SCAFFOLD
                else self.MINI_SWE_AGENT_IMPORT
                if scaffold == self.MINI_SWE_AGENT
                else self.AGENT_IMPORT
            ),
            "--model",
            self._harbor_model(profile),
        ]
        if scaffold in {self.MINI_SWE_AGENT, self.ADVISORY_SCAFFOLD}:
            command.extend(self._mini_swe_agent_args(profile, spec))
        else:
            command.extend(self._budget_shell_args(profile))
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

    def run(self, spec: TrialSpec, *, env_file: str | Path = ".env",
            source_root: Path | None = None) -> int:
        if not self.config.paid_inference_enabled:
            raise RunnerConfigError(
                "paid inference is disabled in the experiment config"
            )
        # Validate the complete plan before staging or dispatch.
        self.command(spec)
        name = self._job_name(spec)
        if (self.config.jobs_root / name).exists() or (self.config.jobs_root / "_poc_records" / name).exists() or (self.config.jobs_root / "_run_records" / name).exists():
            raise RunnerConfigError("trial attempt already exists; use a new attempt number to preserve its records")
        task_dir = self.config.task_root / spec.task_id
        if not task_dir.is_dir():
            raise RunnerConfigError(f"task directory does not exist: {task_dir}")
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
        source_path = str(source_root or Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = (
            source_path + os.pathsep + existing_pythonpath
            if existing_pythonpath
            else source_path
        )
        record_dir = self.config.jobs_root / ("_poc_records" if self._scaffold() == self.ADVISORY_SCAFFOLD else "_run_records") / name
        record_dir.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive host claim protects all scaffolds, including simultaneous
        # invocations racing before Harbor has created its job directory.
        try:
            record_dir.mkdir()
        except FileExistsError:
            raise RunnerConfigError("trial attempt already claimed; use a new attempt") from None
        secrets = tuple(value for key, value in environment.items()
                        if any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "OPENROUTER_KEY")))
        store = RunStore(record_dir, secrets=secrets)
        store_meta = {"task_id": spec.task_id, "profile": spec.profile, "attempt": spec.attempt,
                              "configuration": self.config.resolved_config(),
                              "budget_enforced": self.config.budget_enforced,
                              "provenance": trial_provenance(self.config, task_dir, source_root)}
        store.write_metadata(store_meta)
        started = time.monotonic()
        code = None
        try:
            prepared_task_dir = self._prepare_task(spec, task_dir, expected_sha256=store_meta["provenance"]["task_package_sha256"])
            code = self._dispatch(spec, prepared_task_dir, environment, store)
            return code
        except BaseException as error:
            store.write_summary({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "runner_error",
                                 "exception_type": type(error).__name__, "return_code": None,
                                 "budget_enforced": self.config.budget_enforced, "total_cost_usd": None,
                                 "duration_seconds": time.monotonic() - started})
            raise
        finally:
            if code is not None and self._scaffold() != self.ADVISORY_SCAFFOLD:
                cost = _completed_job_cost(self.config.jobs_root / name / "result.json")
                store.write_summary({"status": "completed" if code == 0 else "runner_error", "return_code": code,
                                     "budget_enforced": self.config.budget_enforced, "total_cost_usd": cost,
                                     "duration_seconds": time.monotonic() - started})

    def _dispatch(self, spec: TrialSpec, prepared_task_dir: Path, environment: dict[str, str], store: RunStore) -> int:
        if self._scaffold() == self.ADVISORY_SCAFFOLD:
            return self._run_poc(spec, prepared_task_dir, environment, store)
        if self._scaffold() != self.MINI_SWE_AGENT:
            completed = run_command(
                self.command(spec, task_dir=prepared_task_dir),
                cwd=self.config.project_root,
                env=environment,
                check=False,
            )
            return completed.returncode

        profile = self.config.profiles[spec.profile]
        job_name = self._job_name(spec)
        reservation_id = f"harbor-mini:{job_name}"
        ledger = BudgetLedger(self.config.budget_db, self.config.budget_cap_usd)
        ledger.reserve(
            reservation_id,
            run_id=job_name,
            model=profile.model,
            max_cost_usd=profile.run_max_cost_usd,
        )
        try:
            completed = run_command(
                self.command(spec, task_dir=prepared_task_dir),
                cwd=self.config.project_root,
                env=environment,
                check=False,
            )
        except BaseException:
            ledger.mark_uncertain(
                reservation_id,
                "Harbor mini-SWE-agent process was interrupted; verify "
                "OpenRouter billing.",
            )
            raise

        cost = _completed_job_cost(self.config.jobs_root / job_name / "result.json")
        if cost is None:
            ledger.mark_uncertain(
                reservation_id,
                "Harbor produced no completed job cost; verify OpenRouter billing.",
            )
        else:
            ledger.reconcile(
                reservation_id,
                cost,
                note="Reconciled from Harbor's completed job result.",
            )
        return completed.returncode

    def _validate_poc(self, spec: TrialSpec) -> None:
        if self.config.budget_enforced:
            raise RunnerConfigError("the advisory POC requires budget.enforce=false")
        if str(self.config.agent.get("version", "")) != "2.4.6":
            raise RunnerConfigError("the advisory POC requires mini-SWE-agent 2.4.6")
        if int(self.config.agent.get("max_steps", 0)) < 1:
            raise RunnerConfigError("the advisory POC requires a positive actor call allowance")
        try:
            policy = AdvicePolicy.load(self.config.consultation)
        except (TypeError, ValueError) as error:
            raise RunnerConfigError(str(error)) from error
        if policy.advice_enabled:
            advisor = self.config.consultation.get("advisor_profile", "frontier")
            if advisor not in self.config.profiles:
                raise RunnerConfigError(f"unknown advisor profile {advisor!r}")
            profile = self.config.profiles[advisor]
            try:
                validate_advisor_routes({"provider": profile.provider, "fallback": profile.fallback})
            except ValueError as error:
                raise RunnerConfigError(str(error)) from error
        manifest = self.config.consultation.get("task_manifest")
        if not manifest:
            raise RunnerConfigError("the SWE POC requires a frozen task_manifest")
        try:
            from .trial_pool import select_tasks
            task_ids = set(select_tasks(None, _under_root(self.config.project_root, manifest)))
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise RunnerConfigError(f"cannot read frozen task manifest: {error}") from error
        if spec.task_id not in task_ids:
            raise RunnerConfigError(f"task {spec.task_id!r} is outside the frozen POC sample")

    def _run_poc(self, spec: TrialSpec, task_dir: Path, environment: dict[str, str], store: RunStore) -> int:
        """Record the run without reserving, settling, or enforcing dollar caps."""
        job_name = self._job_name(spec)
        # Kept outside Harbor's job directory so Harbor can create that directory.
        store.write_metadata({**json.loads((store.run_dir / "metadata.json").read_text()),
            "task_id": spec.task_id, "profile": spec.profile,
            "consultation": self.config.consultation, "budget_enforced": False,
            "actor_max_steps": int(self.config.agent["max_steps"]),
            "configuration": self.config.resolved_config(),
            "harbor_command": self.command(spec, task_dir=task_dir),
        })
        for filename in ("task.toml", "instruction.md"):
            path = task_dir / filename
            if path.is_file():
                artifact = store.write_artifact("task-" + filename.replace(".", "-"), {"content": path.read_text()})
                store.append_event("task_package", {"filename": filename, "artifact": artifact})
        started = time.monotonic()
        store.append_event("dispatch", {"actor_call": 0, "job_name": job_name})
        try:
            with (store.run_dir / "harbor-console.log").open("w") as console:
                completed = run_command(
                    self.command(spec, task_dir=task_dir), cwd=self.config.project_root,
                    env=environment, check=False, stdout=console, stderr=subprocess.STDOUT,
                )
        except BaseException:
            store.write_summary({"status": "interrupted", "budget_enforced": False, "total_cost_usd": None})
            raise
        finally:
            console_path = store.run_dir / "harbor-console.log"
            if console_path.is_file():
                console_path.write_text(store.redact(console_path.read_text(errors="replace")))
        trajectories = list((self.config.jobs_root / job_name).glob("*/agent/mini-swe-agent.trajectory.json"))
        metrics = None
        if len(trajectories) == 1:
            try:
                metrics = json.loads(trajectories[0].read_text())["info"]["poc_metrics"]
            except (OSError, KeyError, TypeError, ValueError):
                pass
        # Grading artifacts are copied only after Harbor exits; they never enter
        # the solving controller or an advisor evidence packet.
        job_dir = self.config.jobs_root / job_name
        for path in sorted(job_dir.glob("*/verifier")):
            destination = store.run_dir / "post_submission" / path.parent.name / "verifier"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(path, destination, copy_function=store.capture_file)
            store.append_event("post_submission_verifier", {"track": "post_submission", "artifact": str(destination.relative_to(store.run_dir))})
        for path in [job_dir / "result.json", *sorted(job_dir.glob("*/result.json"))]:
            if path.is_file():
                artifact = store.write_artifact("harbor-result", {"path": str(path.relative_to(job_dir)), "content": path.read_text()})
                store.append_event("post_submission_result", {"track": "post_submission", "artifact": artifact})
        store.write_summary({
            "status": ("completed" if metrics else "incomplete_records") if completed.returncode == 0 else "runner_error",
            "return_code": completed.returncode, "budget_enforced": False,
            "metrics": metrics, "total_cost_usd": metrics.get("total_cost_usd") if metrics else None,
            "duration_seconds": time.monotonic() - started,
        })
        return completed.returncode

    def _prepare_task(self, spec: TrialSpec, source: Path, *, expected_sha256: str | None = None) -> Path:
        """Stage a task with the scaffold's least-privilege agent network policy."""
        destination = (
            self.config.jobs_root
            / "_prepared_tasks"
            / self._job_name(spec)
        )
        if destination.exists():
            raise RunnerConfigError("prepared attempt already exists; use a new attempt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        if expected_sha256 and package_hash(destination) != expected_sha256:
            raise RunnerConfigError("task package changed during staging; no inference was dispatched")
        manifest = destination / "task.toml"
        if not manifest.is_file():
            raise RunnerConfigError(f"task manifest does not exist: {manifest}")
        source_text = manifest.read_text(encoding="utf-8")
        if self._scaffold() in {self.MINI_SWE_AGENT, self.ADVISORY_SCAFFOLD}:
            source_text = _force_agent_network_policy(
                source_text, "allowlist", ["openrouter.ai"]
            )
        else:
            source_text = _force_agent_network_policy(source_text, "no-network")
        manifest.write_text(source_text, encoding="utf-8")
        return destination

    def _scaffold(self) -> str:
        scaffold = str(self.config.agent.get("scaffold", "budget-shell"))
        if scaffold not in self.SUPPORTED_SCAFFOLDS:
            raise RunnerConfigError(
                f"unsupported agent scaffold {scaffold!r}; expected one of "
                f"{sorted(self.SUPPORTED_SCAFFOLDS)}"
            )
        return scaffold

    def _job_name(self, spec: TrialSpec) -> str:
        if self._scaffold() == self.ADVISORY_SCAFFOLD:
            policy = AdvicePolicy.load(self.config.consultation)
            if policy.phase3:
                return f"phase3-{policy.policy}-m{policy.max_calls}-{spec.profile}-{spec.task_id}-a{spec.attempt}"
            if policy.policy == "advice_requested":
                return f"phase2-advice_requested-m{policy.max_calls}-{spec.profile}-{spec.task_id}-a{spec.attempt}"
            condition = f"{policy.policy}-k{policy.every_steps}" if policy.policy == "fixed_interval" else "actor_only"
            if policy.policy == "fixed_interval" and policy.max_calls != 100:
                condition += f"-m{policy.max_calls}"
            return f"phase1-{condition}-{spec.profile}-{spec.task_id}-a{spec.attempt}"
        return f"{spec.profile}-{spec.task_id}-a{spec.attempt}"

    def _harbor_model(self, profile: ModelProfile) -> str:
        if self._scaffold() in {self.MINI_SWE_AGENT, self.ADVISORY_SCAFFOLD}:
            return f"openrouter/{profile.model}"
        return profile.model

    def _mini_swe_agent_args(self, profile: ModelProfile, spec: TrialSpec) -> list[str]:
        version = str(self.config.agent.get("version", "2.4.6"))
        child_env = {
            # The mini-SWE process needs the key; its bash actions do not. The
            # LocalEnvironment overlay masks credentials from model-run commands.
            "OPENROUTER_API_KEY": "",
            "MSWEA_API_KEY": "",
            "OPENROUTER_KEY": "",
        }
        model_kwargs: dict[str, Any] = {
            "drop_params": True,
        }
        if profile.service_tier:
            model_kwargs["service_tier"] = profile.service_tier
        if profile.provider:
            model_kwargs["extra_body"] = {"provider": profile.provider}
        config = {
            "agent": {"step_limit": int(self.config.agent["max_steps"])},
            "environment": {
                "timeout": int(self.config.agent["command_timeout_sec"]),
                "env": child_env,
            },
            "model": {"model_kwargs": model_kwargs},
        }
        if self._scaffold() == self.ADVISORY_SCAFFOLD:
            advisor_config = {}
            if AdvicePolicy.load(self.config.consultation).advice_enabled:
                advisor = self.config.profiles[self.config.consultation.get("advisor_profile", "frontier")]
                advisor_config = {
                    "model": advisor.model, "reasoning_effort": advisor.reasoning_effort,
                    "service_tier": advisor.service_tier, "provider": advisor.provider,
                    "fallback": advisor.fallback,
                }
            config["agent"].update({
                "agent_class": "autoresearch_budget.advisory_agent.AdvisoryAgent",
                "max_observation_chars": int(self.config.agent["max_observation_chars"]),
                "advice": {
                    **{key: self.config.consultation[key] for key in AdvicePolicy.__dataclass_fields__
                       if key in self.config.consultation},
                    "advisor": advisor_config,
                    "run_id": self._job_name(spec),
                },
            })
            config["model"]["model_class"] = "autoresearch_budget.actor_model.StructuredActorModel"
            config["environment"]["environment_class"] = "autoresearch_budget.poc_environment.PocEnvironment"
        args = [
            "--ak",
            f"version={version}",
            "--ak",
            f"cost_limit={profile.run_max_cost_usd if self.config.budget_enforced else 0}",
            "--ak",
            f"max_tokens={int(self.config.agent['max_tokens_per_call'])}",
            "--ak",
            "config=" + json.dumps(config, separators=(",", ":")),
        ]
        if profile.reasoning_effort:
            args.extend(["--ak", f"reasoning_effort={profile.reasoning_effort}"])
        return args

    def _budget_shell_args(self, profile: ModelProfile) -> list[str]:
        args = [
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
            args.extend(["--ak", f"reasoning_effort={profile.reasoning_effort}"])
        if profile.provider:
            args.extend(
                [
                    "--ak",
                    "provider=" + json.dumps(profile.provider, separators=(",", ":")),
                ]
            )
        return args

    def _harbor_prefix(self) -> list[str]:
        prefix = [sys.executable, "-m", "autoresearch_budget.docker_runtime"]
        explicit = os.environ.get("HARBOR_BIN")
        if explicit:
            return [*prefix, explicit]
        # Invoke Harbor in this interpreter, where the harness is installed.
        return [*prefix, sys.executable, "-m", "harbor.cli.main"]



def _directory_name(value) -> bool:
    return (isinstance(value, str) and bool(value.strip()) and value == value.strip()
            and value not in {".", ".."} and "/" not in value and "\\" not in value
            and not any(ord(char) < 32 for char in value))


def _validate_config(raw) -> None:
    if not isinstance(raw, dict):
        raise RunnerConfigError("configuration must be a JSON object")
    for name in ("budget", "agent", "profiles"):
        if not isinstance(raw.get(name), dict) or not raw[name]:
            raise RunnerConfigError(f"{name} must be a nonempty object")
    for name in ("locations", "consultation"):
        if name in raw and not isinstance(raw[name], dict):
            raise RunnerConfigError(f"{name} must be an object")
    budget, agent = raw["budget"], raw["agent"]
    for name, value in (("paid_inference_enabled", raw.get("paid_inference_enabled", False)),
                        ("budget.enforce", budget.get("enforce", True))):
        if type(value) is not bool:
            raise RunnerConfigError(f"{name} must be a JSON boolean")
    enforced = budget.get("enforce", True)
    def money(name, value):
        try:
            amount = Decimal(str(value))
        except (ValueError, InvalidOperation):
            raise RunnerConfigError(f"{name} must be a finite nonnegative amount") from None
        if not amount.is_finite() or amount < 0:
            raise RunnerConfigError(f"{name} must be a finite nonnegative amount")
    if enforced and "cap_usd" not in budget:
        raise RunnerConfigError("budget.cap_usd is required for enforced accounting")
    money("budget.cap_usd", budget.get("cap_usd", 0))
    for name, value in [("budget.ledger", budget.get("ledger")),
                        *[("locations." + key, value) for key, value in raw.get("locations", {}).items()],
                        *([( "workspace", raw["workspace"])] if "workspace" in raw else [])]:
        if not isinstance(value, str) or not value.strip():
            raise RunnerConfigError(f"{name} must be a nonempty path string")
    for name in ("max_steps", "max_tokens_per_call", "command_timeout_sec", "max_observation_chars"):
        value = agent.get(name)
        if type(value) is not int or value < (100 if name == "max_observation_chars" else 1):
            raise RunnerConfigError(f"agent.{name} must be a positive integer" + (" of at least 100" if name == "max_observation_chars" else ""))
    if agent.get("scaffold", "budget-shell") not in HarborTrialRunner.SUPPORTED_SCAFFOLDS:
        raise RunnerConfigError("unsupported agent scaffold")
    if agent.get("scaffold") in {"mini-swe-agent", "mini-swe-advisor"} and str(agent.get("version")) != "2.4.6":
        raise RunnerConfigError("mini-SWE execution requires version 2.4.6")
    for name, profile in raw["profiles"].items():
        if not _directory_name(name) or not isinstance(profile, dict):
            raise RunnerConfigError("profiles need plain names and object values")
        if not isinstance(profile.get("model"), str) or not profile["model"].strip():
            raise RunnerConfigError(f"profiles.{name}.model must be a nonempty string")
        if enforced and "run_max_cost_usd" not in profile:
            raise RunnerConfigError(f"profiles.{name}.run_max_cost_usd is required")
        money(f"profiles.{name}.run_max_cost_usd", profile.get("run_max_cost_usd", 0))
        for key in ("provider", "fallback"):
            if profile.get(key) is not None and not isinstance(profile[key], dict):
                raise RunnerConfigError(f"profiles.{name}.{key} must be an object")
    if not enforced and agent.get("scaffold") != "mini-swe-advisor":
        raise RunnerConfigError("budget.enforce=false requires mini-swe-advisor")
    if agent.get("scaffold") == "mini-swe-advisor":
        if enforced:
            raise RunnerConfigError("advisory execution requires budget.enforce=false")
        try:
            AdvicePolicy.load(raw.get("consultation", {}))
        except (TypeError, ValueError) as error:
            raise RunnerConfigError(str(error)) from error


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


def _force_agent_network_policy(
    source: str, mode: str, allowed_hosts: list[str] | None = None
) -> str:
    """Set Harbor's agent-phase network policy without rewriting unrelated TOML."""
    if mode not in {"no-network", "allowlist", "public"}:
        raise RunnerConfigError(f"unsupported network mode {mode!r}")
    hosts = list(allowed_hosts or [])
    if mode != "allowlist" and hosts:
        raise RunnerConfigError("allowed hosts require allowlist mode")
    if mode == "allowlist" and not hosts:
        raise RunnerConfigError("allowlist mode requires at least one host")
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
    filtered = [
        line
        for line in lines[start + 1 : end]
        if line.split("=", 1)[0].strip() not in {"network_mode", "allowed_hosts"}
    ]
    policy = [f'network_mode = "{mode}"\n']
    if hosts:
        policy.append(
            "allowed_hosts = " + json.dumps(hosts, separators=(",", ":")) + "\n"
        )
    lines[start + 1 : end] = [*policy, *filtered]
    return "".join(lines)


def _completed_job_cost(path: Path) -> Decimal | None:
    """Read billed OpenRouter cost from a finished one-trial Harbor job.

    Newly released models can be absent from LiteLLM's local price table, making
    Harbor's aggregate incorrectly report zero. The raw OpenRouter responses in
    mini-SWE's trajectory include the billed cost and take precedence when every
    recorded API call has one.
    """
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
        if not result.get("finished_at") or result.get("n_total_trials") != 1:
            return None
        billed = _trajectory_billed_cost(path.parent)
        if billed is not None:
            return billed
        raw_cost = result["stats"]["cost_usd"]
        cost = Decimal(str(raw_cost))
        if not cost.is_finite() or cost < 0:
            return None
        return cost
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        InvalidOperation,
        json.JSONDecodeError,
    ):
        return None


def _trajectory_billed_cost(job_dir: Path) -> Decimal | None:
    trajectories = list(job_dir.glob("*/agent/mini-swe-agent.trajectory.json"))
    if len(trajectories) != 1:
        return None
    try:
        trajectory = json.loads(trajectories[0].read_text(encoding="utf-8"))
        expected = int(trajectory["info"]["model_stats"]["api_calls"])
        costs: list[Decimal] = []
        for message in trajectory["messages"]:
            response = message.get("extra", {}).get("response") or {}
            usage = response.get("usage") or {}
            if "cost" not in usage or usage["cost"] is None:
                continue
            cost = Decimal(str(usage["cost"]))
            if not cost.is_finite() or cost < 0:
                return None
            costs.append(cost)
        if expected < 1 or len(costs) != expected:
            return None
        return sum(costs, Decimal("0"))
    except (OSError, KeyError, TypeError, ValueError, InvalidOperation, json.JSONDecodeError):
        return None
