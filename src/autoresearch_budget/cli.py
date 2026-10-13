"""Local control commands that never call a model implicitly."""

from __future__ import annotations

import argparse
import asyncio
import json
import hashlib
import sqlite3
import shlex
import tempfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from .agent import AgentLimits, ExecResult, MinimalShellAgent
from .budget import BudgetError, BudgetLedger
from .models import ScriptedModel
from .probe import run_probe
from .runner import ExperimentConfig, HarborTrialRunner, RunnerConfigError, TrialSpec
from .store import RunStore
from .batch import BatchRunner
from .trial_pool import MAX_WORKERS, TrialPool, select_tasks
from .advisor import POLICIES


class _MockSandbox:
    def __init__(self):
        self.commands: list[str] = []

    async def exec(self, command: str, *, timeout_sec: int) -> ExecResult:
        del timeout_sec
        self.commands.append(command)
        return ExecResult(stdout="README.md\n", stderr="", return_code=0)


async def _mock_run(output: Path) -> dict[str, object]:
    store = RunStore(output)
    model = ScriptedModel(
        [
            '{"action":"shell","command":"git status --short"}',
            '{"action":"finish","summary":"Mock run completed."}',
        ]
    )
    sandbox = _MockSandbox()
    agent = MinimalShellAgent(
        model=model,
        sandbox=sandbox,
        store=store,
        run_id="mock-run",
        limits=AgentLimits(
            max_steps=3, max_tokens_per_call=128, command_timeout_sec=5
        ),
    )
    store.write_metadata({"network": False, "model": "mock/scripted"})
    outcome = await agent.run("Inspect the repository and report status.")
    return {"outcome": asdict(outcome), "commands": sandbox.commands}


def _advice_arguments(command):
    command.add_argument("--advice-policy", choices=sorted(POLICIES), help="override consultation policy")
    command.add_argument("--max-advice-calls", type=int, help="maximum logical consultations per task attempt; zero removes advice tools and reviews while retaining the policy's planning/checkpoints")
    command.add_argument("--advice-every-steps", type=int, help="actor-call interval; fixed_interval only")


def _load_experiment(args):
    return ExperimentConfig.load(args.config).with_consultation_overrides(
        policy=args.advice_policy, max_calls=args.max_advice_calls, every_steps=args.advice_every_steps)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch-budget")
    commands = parser.add_subparsers(dest="command", required=True)

    from .onboarding import PRESETS
    init = commands.add_parser("init", help="create a workspace from a disabled preset")
    init.add_argument("--workspace", type=Path, default=Path.cwd())
    init.add_argument("--preset", choices=PRESETS, default="hybrid")
    doctor = commands.add_parser("doctor", help="check the environment without inference")
    doctor.add_argument("--config", type=Path)
    doctor.add_argument("--offline", action="store_true", help="skip Docker, task packages and credentials")
    doctor.add_argument("--env-file", default=".env")
    demo = commands.add_parser("demo", help="demonstrate the current advisory controller without network")
    demo.add_argument("--output", type=Path)
    dataset = commands.add_parser("dataset", help="validate existing external Harbor task packages")
    datasets = dataset.add_subparsers(dest="dataset_command", required=True)
    inspect = datasets.add_parser("inspect")
    inspect.add_argument("--task-root", type=Path, required=True)
    inspect.add_argument("--task-manifest", type=Path, required=True)
    inspect.add_argument("--output", type=Path, help="save package checksums to a new JSON file")

    report = commands.add_parser("report", help="summarize saved attempts without execution")
    report.add_argument("source", type=Path)
    report.add_argument("--format", choices=("text", "json", "csv"), default="text")
    report.add_argument("--output", type=Path)
    report.add_argument("--budget-db", type=Path, help="attach the current standalone ledger snapshot")
    comparison = commands.add_parser("compare", help="compare matched local task outcomes")
    comparison.add_argument("left", type=Path)
    comparison.add_argument("right", type=Path)
    comparison.add_argument("--output", type=Path)

    budget = commands.add_parser("budget", help="manage the inference ledger")
    budget_commands = budget.add_subparsers(dest="budget_command", required=True)
    for name in ("init", "show", "reservations", "reconcile", "correct"):
        command = budget_commands.add_parser(name)
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--cap-usd", type=Decimal, required=name == "init")
        if name in {"reconcile", "correct"}:
            command.add_argument("--reservation-id", required=True)
            command.add_argument("--actual-cost-usd", type=Decimal, required=True)
            command.add_argument("--evidence", type=Path, required=True, help="provider billing or final cost evidence file")
            command.add_argument("--note", required=True)

    mock = commands.add_parser("mock-run", help="zero-network scaffold smoke test")
    mock.add_argument("--output", type=Path)

    trial = commands.add_parser("trial", help="plan or run Harbor trials with up to eight workers")
    trial_commands = trial.add_subparsers(dest="trial_command", required=True)
    for name in ("plan", "run"):
        command = trial_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--profile", required=True)
        command.add_argument("--json", action="store_true", help="include resolved accounting and task selection in a JSON plan")
        selection = command.add_mutually_exclusive_group(required=True)
        selection.add_argument("--task-id", action="append", help="task ID; repeat to select multiple tasks")
        selection.add_argument("--task-manifest", type=Path, help="JSON manifest containing tasks[].task_id")
        command.add_argument("--workers", type=int, choices=range(1, MAX_WORKERS + 1), default=1)
        command.add_argument("--env-file", default=".env")
        command.add_argument("--attempt", type=int, default=1)
        _advice_arguments(command)

    probe = commands.add_parser("probe", help="run one low-cost model probe")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--profile", required=True)
    probe.add_argument("--max-cost-usd", type=Decimal, required=True)
    probe.add_argument("--max-tokens", type=int, default=128)
    batch = commands.add_parser("batch", help="preview or execute an advisor experiment cohort")
    batch_commands = batch.add_subparsers(dest="batch_command", required=True)
    for name in ("plan", "run"):
        command = batch_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--batch-dir", type=Path, required=True)
        command.add_argument("--profile", default="small")
        command.add_argument("--concurrency", type=int, default=4)
        command.add_argument("--env-file", default=".env")
        _advice_arguments(command)
    status = batch_commands.add_parser("status", help="read progress for every active task")
    status.add_argument("--batch-dir", type=Path, required=True)
    status.add_argument("--live", action="store_true", help="also read live advice logs via Docker")
    drain = batch_commands.add_parser("drain", help="stop new dispatch while active attempts finish")
    drain.add_argument("--batch-dir", type=Path, required=True)
    drain.add_argument("--clear", action="store_true", help="remove the drain request before resuming")
    return parser


def main(argv: list[str] | None = None) -> int:
    from .processes import cancellation_scope
    try:
        with cancellation_scope():
            return _main(argv)
    except KeyboardInterrupt:
        print("interrupted: inspect attempt records, Docker state and billing before resuming")
        return 130


def _main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command in {"report", "compare"}:
        try:
            from .reporting import report, compare, render
            base = args.output.parent if args.output else None
            result = report(args.source, link_root=base) if args.command == "report" else compare(report(args.left), report(args.right))
            if args.command == "report" and args.budget_db:
                from .budget import inspect_ledger
                result["budget"] = inspect_ledger(args.budget_db)
            content = render(result, args.format if args.command == "report" else "json")
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("x") as stream:
                    stream.write(content)
                print(str(args.output.resolve()))
            else:
                print(content, end="")
            return 0
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, BudgetError) as error:
            print(f"error: {error}")
            return 2
    if args.command in {"init", "doctor", "demo", "dataset"}:
        try:
            if args.command == "init":
                from .onboarding import initialize
                result = initialize(args.workspace, args.preset)
            elif args.command == "doctor":
                from .onboarding import doctor
                result = doctor(args.config, offline=args.offline, env_file=args.env_file)
            elif args.command == "dataset":
                from .dataset import inspect_tasks
                result = inspect_tasks(args.task_root, args.task_manifest)
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open("x") as stream:
                        json.dump(result, stream, indent=2)
            else:
                from .demo import run_demo
                if args.output:
                    result = run_demo(args.output)
                else:
                    with tempfile.TemporaryDirectory(prefix="autoresearch-demo-output-") as temporary:
                        result = run_demo(Path(temporary))
                    result["output"] = None
            print(json.dumps(result, indent=2))
            return 1 if args.command == "doctor" and not result["ok"] else 0
        except ImportError:
            print("error: install autoresearch-budget[runtime] for the advisory demo")
            return 2
        except (OSError, ValueError, TypeError, KeyError) as error:
            print(f"error: {error}")
            return 2
    if args.command == "batch":
        try:
            if args.batch_command == "drain":
                if not (args.batch_dir / "metadata.json").is_file():
                    raise ValueError("batch metadata is missing; select an initialized batch")
                flag = args.batch_dir / "drain.request"
                if args.clear:
                    flag.unlink(missing_ok=True)
                else:
                    flag.touch(exist_ok=True)
                print(json.dumps({"batch_dir": str(args.batch_dir.resolve()), "drain_requested": not args.clear}))
                return 0
            if args.batch_command == "status":
                from .batch_monitor import snapshot
                print(json.dumps(snapshot(args.batch_dir, live=args.live), indent=2))
                return 0
            config = _load_experiment(args)
            runner = BatchRunner(config, args.batch_dir, profile=args.profile,
                                 concurrency=args.concurrency, env_file=args.env_file)
            if args.batch_command == "plan":
                preview = runner.plan()
                preview["attempts"] = [{key: record.get(key) for key in
                                       ("task_id", "attempt", "job_name", "return_code")}
                                      for record in preview.pop("records")]
                print(json.dumps(preview, indent=2))
                return 0
            return runner.run()
        except (RunnerConfigError, OSError, ValueError, KeyError, TypeError) as error:
            # Parser/configuration failures must not accidentally launch paid work.
            print(f"error: {error}")
            return 2
    if args.command == "budget":
        try:
            from .budget import inspect_ledger
            if args.budget_command == "init":
                ledger = BudgetLedger(args.db, args.cap_usd)
                result = asdict(ledger.snapshot())
            else:
                result = inspect_ledger(args.db)
                cap = Decimal(result["snapshot"]["cap_usd"])
                if args.cap_usd is not None and args.cap_usd != cap:
                    raise ValueError("use the ledger's original cap")
                if args.budget_command in {"reconcile", "correct"}:
                    if not args.note.strip():
                        raise ValueError("a nonempty reconciliation note is required")
                    digest = hashlib.sha256(args.evidence.read_bytes()).hexdigest()
                    note = f"{args.note}; evidence={args.evidence.resolve()}; sha256={digest}"
                    ledger = BudgetLedger(args.db, cap)
                    method = ledger.correct_settled if args.budget_command == "correct" else ledger.reconcile
                    method(args.reservation_id, args.actual_cost_usd, note=note)
                    result = inspect_ledger(args.db)
                elif args.budget_command == "show":
                    result = result["snapshot"]
            print(json.dumps(result, default=str, indent=2))
            return 0
        except (BudgetError, OSError, ValueError, sqlite3.Error) as error:
            print(f"error: {error}")
            return 2
    if args.command == "mock-run":
        if args.output:
            output = args.output
            result = asyncio.run(_mock_run(output))
        else:
            with tempfile.TemporaryDirectory(prefix="autoresearch-mock-") as temp:
                result = asyncio.run(_mock_run(Path(temp)))
        print(json.dumps(result, default=str, indent=2))
        return 0
    if args.command == "trial":
        try:
            config = _load_experiment(args)
            runner = HarborTrialRunner(config)
            tasks = select_tasks(args.task_id, args.task_manifest)
            specs = [TrialSpec(task, args.profile, args.attempt) for task in tasks]
            pool = TrialPool(runner, specs, workers=args.workers, env_file=args.env_file)
            if args.trial_command == "plan":
                if len(specs) == 1 and not args.json:
                    print(shlex.join(pool.commands[0]))
                else:
                    print(json.dumps(pool.plan(), indent=2))
                return 0
            if len(specs) == 1:
                return runner.run(specs[0], env_file=args.env_file)
            return pool.run(report=lambda result: print(json.dumps(result), flush=True))
        except (RunnerConfigError, BudgetError, OSError, ValueError, KeyError, TypeError) as error:
            print(f"error: {error}")
            return 2
    if args.command == "probe":
        try:
            config = ExperimentConfig.load(args.config)
            result = asyncio.run(
                run_probe(
                    config,
                    profile_name=args.profile,
                    max_cost_usd=args.max_cost_usd,
                    max_tokens=args.max_tokens,
                )
            )
            print(json.dumps(asdict(result), default=str, indent=2))
            return 0 if result.action_format_valid else 1
        except Exception as error:
            print(f"error: {type(error).__name__}: {error}")
            return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
