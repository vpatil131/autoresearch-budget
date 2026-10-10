"""Local control commands that never call a model implicitly."""

from __future__ import annotations

import argparse
import asyncio
import json
import shlex
import tempfile
from dataclasses import asdict
from decimal import Decimal
from pathlib import Path

from .agent import AgentLimits, ExecResult, MinimalShellAgent
from .budget import BudgetLedger
from .models import ScriptedModel
from .probe import run_probe
from .runner import ExperimentConfig, HarborTrialRunner, RunnerConfigError, TrialSpec
from .store import RunStore
from .batch import BatchRunner


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoresearch-budget")
    commands = parser.add_subparsers(dest="command", required=True)

    budget = commands.add_parser("budget", help="manage the inference ledger")
    budget_commands = budget.add_subparsers(dest="budget_command", required=True)
    for name in ("init", "show"):
        command = budget_commands.add_parser(name)
        command.add_argument("--db", type=Path, required=True)
        command.add_argument("--cap-usd", type=Decimal, required=True)

    mock = commands.add_parser("mock-run", help="zero-network scaffold smoke test")
    mock.add_argument("--output", type=Path)

    trial = commands.add_parser("trial", help="plan or run one Harbor model trial")
    trial_commands = trial.add_subparsers(dest="trial_command", required=True)
    for name in ("plan", "run"):
        command = trial_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--profile", required=True)
        command.add_argument("--task-id", required=True)
        command.add_argument("--attempt", type=int, default=1)

    probe = commands.add_parser("probe", help="run one low-cost model probe")
    probe.add_argument("--config", type=Path, required=True)
    probe.add_argument("--profile", required=True)
    probe.add_argument("--max-cost-usd", type=Decimal, required=True)
    probe.add_argument("--max-tokens", type=int, default=128)
    batch = commands.add_parser("batch", help="preview or execute the phase 1 task cohort")
    batch_commands = batch.add_subparsers(dest="batch_command", required=True)
    for name in ("plan", "run"):
        command = batch_commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--batch-dir", type=Path, required=True)
        command.add_argument("--profile", default="small")
        command.add_argument("--concurrency", type=int, default=4)
        command.add_argument("--env-file", default=".env")
    status = batch_commands.add_parser("status", help="read progress for every active task")
    status.add_argument("--batch-dir", type=Path, required=True)
    status.add_argument("--live", action="store_true", help="also read live advice logs via Docker")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "batch":
        try:
            if args.batch_command == "status":
                from .batch_monitor import snapshot
                print(json.dumps(snapshot(args.batch_dir, live=args.live), indent=2))
                return 0
            config = ExperimentConfig.load(args.config, project_root=Path.cwd())
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
        except (RunnerConfigError, OSError, ValueError, KeyError) as error:
            # Parser/configuration failures must not accidentally launch paid work.
            print(f"error: {error}")
            return 2
    if args.command == "budget":
        ledger = BudgetLedger(args.db, args.cap_usd)
        print(json.dumps(asdict(ledger.snapshot()), default=str, indent=2))
        return 0
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
            config = ExperimentConfig.load(args.config, project_root=Path.cwd())
            runner = HarborTrialRunner(config)
            spec = TrialSpec(args.task_id, args.profile, args.attempt)
            if args.trial_command == "plan":
                print(shlex.join(runner.command(spec)))
                return 0
            return runner.run(spec)
        except RunnerConfigError as error:
            print(f"error: {error}")
            return 2
    if args.command == "probe":
        try:
            config = ExperimentConfig.load(args.config, project_root=Path.cwd())
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
