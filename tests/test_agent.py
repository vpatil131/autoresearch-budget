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
from autoresearch_budget.models import Completion, ModelError, ScriptedModel
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

    def test_action_parser_accepts_one_object_after_brief_reasoning(self):
        self.assertEqual(
            parse_action(
                'I will inspect the file.\n{"action":"shell","command":"sed -n \'1,80p\' app.py"}'
            ),
            {"action": "shell", "command": "sed -n '1,80p' app.py"},
        )

    def test_action_parser_rejects_multiple_embedded_actions(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            parse_action(
                '{"action":"shell","command":"one"}\n'
                '{"action":"shell","command":"two"}'
            )

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

    def test_command_timeout_becomes_an_agent_observation(self):
        class TimedOutSandbox:
            async def exec(self, command: str, *, timeout_sec: int) -> ExecResult:
                raise TimeoutError(
                    f"Command timed out after {timeout_sec} seconds: {command}"
                )

        with TemporaryDirectory() as temp:
            model = ScriptedModel(
                [
                    '{"action":"shell","command":"slow test"}',
                    '{"action":"finish","summary":"used a narrower check"}',
                ]
            )
            agent = MinimalShellAgent(
                model=model,
                sandbox=TimedOutSandbox(),
                store=RunStore(temp),
                run_id="timeout-test",
                limits=AgentLimits(2, 100, 7),
            )
            outcome = asyncio.run(agent.run("Fix it"))
            self.assertEqual(outcome.status, "finished")
            self.assertIn("exit code: 124", model.requests[1][-1]["content"])
            self.assertIn("timed out after 7 seconds", model.requests[1][-1]["content"])
            records = [
                json.loads(line)
                for line in (Path(temp) / "events.jsonl").read_text().splitlines()
            ]
            self.assertEqual(records[1]["payload"]["return_code"], 124)

    def test_explicit_model_error_preserves_prior_known_cost(self):
        class OneCompletionThenError:
            def __init__(self):
                self.calls = 0

            async def complete(self, messages, *, request_id, max_tokens):
                del messages, max_tokens
                self.calls += 1
                if self.calls == 2:
                    raise ModelError("request rejected")
                return Completion(
                    content='{"action":"shell","command":"edit"}',
                    response_id=request_id,
                    requested_model="mock/model",
                    served_model="mock/model",
                    cost_usd=Decimal("0.03"),
                )

        with TemporaryDirectory() as temp:
            agent = MinimalShellAgent(
                model=OneCompletionThenError(),
                sandbox=RecordingSandbox(),
                store=RunStore(temp),
                run_id="model-error-test",
                limits=AgentLimits(3, 100, 7),
            )
            outcome = asyncio.run(agent.run("Fix it"))
            self.assertEqual(outcome.status, "model_error")
            self.assertEqual(outcome.total_cost_usd, Decimal("0.03"))
            self.assertEqual(outcome.steps, 2)


if __name__ == "__main__":
    unittest.main()
