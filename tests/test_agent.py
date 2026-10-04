import asyncio
import json
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from autoresearch_budget.agent import (
    AgentLimits,
    ExecResult,
    MinimalShellAgent,
    parse_action,
)
from autoresearch_budget.models import Completion, ScriptedModel
from autoresearch_budget.store import RunStore


class RecordingSandbox:
    def __init__(self):
        self.commands = []

    async def exec(self, command: str, *, timeout_sec: int) -> ExecResult:
        self.commands.append((command, timeout_sec))
        return ExecResult(stdout="ok", stderr="", return_code=0)


class MinimalShellAgentTest(unittest.TestCase):
    def test_shell_then_finish_writes_complete_trajectory(self):
        with TemporaryDirectory() as temp:
            sandbox = RecordingSandbox()
            model = ScriptedModel(
                [
                    '```json\n{"action":"shell","command":"pwd"}\n```',
                    '{"action":"finish","summary":"done"}',
                ]
            )
            agent = MinimalShellAgent(
                model=model,
                sandbox=sandbox,
                store=RunStore(temp),
                run_id="test-run",
                limits=AgentLimits(3, 100, 7),
            )
            outcome = asyncio.run(agent.run("Fix it"))
            self.assertEqual(outcome.status, "finished")
            self.assertEqual(outcome.steps, 2)
            self.assertEqual(sandbox.commands, [("pwd", 7)])
            records = [
                json.loads(line)
                for line in (Path(temp) / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                [record["kind"] for record in records],
                ["model_response", "shell_result", "model_response"],
            )
            self.assertTrue((Path(temp) / "summary.json").exists())

    def test_malformed_action_becomes_observation(self):
        with TemporaryDirectory() as temp:
            model = ScriptedModel(
                ["not json", '{"action":"finish","summary":"recovered"}']
            )
            agent = MinimalShellAgent(
                model=model,
                sandbox=RecordingSandbox(),
                store=RunStore(temp),
                run_id="test-run",
                limits=AgentLimits(2, 100, 7),
            )
            outcome = asyncio.run(agent.run("Fix it"))
            self.assertEqual(outcome.status, "finished")
            self.assertIn("Invalid action", model.requests[1][-1]["content"])

    def test_action_parser_rejects_ambiguous_shape(self):
        with self.assertRaises(ValueError):
            parse_action('{"action":"shell","command":""}')

    def test_invalid_limits_fail_before_a_run(self):
        with TemporaryDirectory() as temp:
            with self.assertRaises(ValueError):
                MinimalShellAgent(
                    model=ScriptedModel([]),
                    sandbox=RecordingSandbox(),
                    store=RunStore(temp),
                    run_id="invalid-limits",
                    limits=AgentLimits(0, 100, 7),
                )

    def test_cost_envelope_stops_before_shell_action(self):
        class CostedModel:
            async def complete(self, messages, *, request_id, max_tokens):
                del messages, max_tokens
                return Completion(
                    content='{"action":"shell","command":"expensive-step"}',
                    response_id=request_id,
                    requested_model="mock/costed",
                    served_model="mock/costed",
                    cost_usd=Decimal("0.05"),
                )

        with TemporaryDirectory() as temp:
            sandbox = RecordingSandbox()
            agent = MinimalShellAgent(
                model=CostedModel(),
                sandbox=sandbox,
                store=RunStore(temp),
                run_id="cost-limited",
                limits=AgentLimits(3, 100, 7, Decimal("0.05")),
            )
            outcome = asyncio.run(agent.run("Fix it"))
            self.assertEqual(outcome.status, "cost_limit")
            self.assertEqual(sandbox.commands, [("expensive-step", 7)])

    def test_large_shell_output_is_truncated_in_model_context_but_stored(self):
        class LargeOutputSandbox:
            async def exec(self, command: str, *, timeout_sec: int) -> ExecResult:
                del command, timeout_sec
                return ExecResult(stdout="x" * 100, stderr="", return_code=0)

        with TemporaryDirectory() as temp:
            model = ScriptedModel(
                [
                    '{"action":"shell","command":"make noise"}',
                    '{"action":"finish","summary":"done"}',
                ]
            )
            agent = MinimalShellAgent(
                model=model,
                sandbox=LargeOutputSandbox(),
                store=RunStore(temp),
                run_id="truncation-test",
                limits=AgentLimits(2, 100, 7, None, 20),
            )
            asyncio.run(agent.run("Fix it"))
            self.assertIn("80 characters omitted", model.requests[1][-1]["content"])
            events = (Path(temp) / "events.jsonl").read_text()
            self.assertIn("x" * 100, events)


if __name__ == "__main__":
    unittest.main()
