"""A minimal, model-neutral shell agent scaffold."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from decimal import Decimal
from typing import Protocol

from .models import CompletionBackend, ModelError, RequestOutcomeUnknown
from .store import RunStore


SYSTEM_PROMPT = """You are a software engineering agent working in a disposable repository.
Solve the user's issue by inspecting files, editing code, and running visible tests.

Return exactly one JSON object per turn, with no prose outside it:
{"action":"shell","command":"one shell command"}
or
{"action":"finish","summary":"what you changed and validated"}

The shell command runs in the repository working directory. Never attempt to read
benchmark solutions, hidden tests, verifier files, credentials, or host files.
The task environment has no network access during your work. Do not attempt to
download packages or query public services. If a visible test cannot run because
an uncached dependency is unavailable, continue with repository-local inspection
and static validation instead of retrying the download.
Work in tracked repository source, not vendored dependencies or node_modules.
Do not run package installers or modify lockfiles unless the issue explicitly
requires a dependency change. Locate the relevant source quickly, implement a
focused fix, and reserve the final quarter of the step budget for tests, diff
review, and correction. Batch related read-only inspection in one shell action
when practical.
Finish only after checking the diff and running relevant visible tests when feasible.
"""


@dataclass(frozen=True)
class ExecResult:
    stdout: str
    stderr: str
    return_code: int


class Sandbox(Protocol):
    async def exec(self, command: str, *, timeout_sec: int) -> ExecResult: ...


@dataclass(frozen=True)
class AgentLimits:
    max_steps: int
    max_tokens_per_call: int
    command_timeout_sec: int
    max_total_cost_usd: Decimal | None = None
    max_observation_chars: int = 20_000


@dataclass(frozen=True)
class AgentOutcome:
    status: str
    steps: int
    total_cost_usd: Decimal | None
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    summary: str


class MinimalShellAgent:
    def __init__(
        self,
        *,
        model: CompletionBackend,
        sandbox: Sandbox,
        store: RunStore,
        run_id: str,
        limits: AgentLimits,
    ):
        if limits.max_steps < 1:
            raise ValueError("max_steps must be positive")
        if limits.max_tokens_per_call < 1:
            raise ValueError("max_tokens_per_call must be positive")
        if limits.command_timeout_sec < 1:
            raise ValueError("command_timeout_sec must be positive")
        if limits.max_observation_chars < 1:
            raise ValueError("max_observation_chars must be positive")
        self.model = model
        self.sandbox = sandbox
        self.store = store
        self.run_id = run_id
        self.limits = limits

    async def run(self, instruction: str) -> AgentOutcome:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]
        total_cost = Decimal("0")
        cost_known = True
        token_totals = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_tokens": 0,
            "reasoning_tokens": 0,
        }
        status = "step_limit"
        summary = "Agent reached its step limit."

        for step in range(1, self.limits.max_steps + 1):
            request_id = f"{self.run_id}:step:{step}"
            try:
                completion = await self.model.complete(
                    messages,
                    request_id=request_id,
                    max_tokens=self.limits.max_tokens_per_call,
                )
            except RequestOutcomeUnknown:
                self.store.append_event(
                    "model_outcome_unknown", {"step": step, "request_id": request_id}
                )
                raise
            except ModelError as error:
                # An explicit provider rejection has a known outcome: no
                # completion was accepted for this step. Preserve the patch and
                # reconcile prior completed calls instead of converting their
                # whole reservation into uncertain spend.
                status = "model_error"
                summary = f"Model request failed before a completion: {error}"
                self.store.append_event(
                    "model_error",
                    {"step": step, "request_id": request_id, "error": str(error)},
                )
                break

            if completion.cost_usd is None:
                cost_known = False
            elif cost_known:
                total_cost += completion.cost_usd
            for field in token_totals:
                token_totals[field] += int(getattr(completion, field))
            self.store.append_event(
                "model_response",
                {
                    "step": step,
                    "request_id": request_id,
                    "response_id": completion.response_id,
                    "requested_model": completion.requested_model,
                    "served_model": completion.served_model,
                    "content": completion.content,
                    "cost_usd": completion.cost_usd,
                    "usage": completion.raw_usage,
                    "provider_metadata": completion.provider_metadata,
                    "finish_reason": completion.finish_reason,
                },
            )
            messages.append({"role": "assistant", "content": completion.content})
            stop_after_action: tuple[str, str] | None = None
            if not cost_known:
                stop_after_action = (
                    "cost_unknown",
                    "Run stopped because the response omitted billed usage.cost.",
                )
            elif (
                self.limits.max_total_cost_usd is not None
                and total_cost >= self.limits.max_total_cost_usd
            ):
                stop_after_action = (
                    "cost_limit",
                    f"Run stopped after reaching its ${self.limits.max_total_cost_usd} "
                    "model-cost envelope.",
                )

            try:
                action = parse_action(completion.content)
            except ValueError as error:
                observation = f"Invalid action: {error}"
                self.store.append_event(
                    "invalid_action", {"step": step, "error": str(error)}
                )
                messages.append({"role": "user", "content": observation})
                if stop_after_action:
                    status, summary = stop_after_action
                    self.store.append_event(
                        status, {"step": step, "total_cost_usd": total_cost}
                    )
                    break
                continue

            if action["action"] == "finish":
                status = "finished"
                summary = action["summary"]
                break

            try:
                result = await self.sandbox.exec(
                    action["command"], timeout_sec=self.limits.command_timeout_sec
                )
            except TimeoutError as error:
                # A slow repository command is an observation about the task, not
                # an infrastructure failure. Let the model choose a narrower test,
                # inspect the partial work, or finish within its remaining steps.
                result = ExecResult(
                    stdout="",
                    stderr=str(error),
                    return_code=124,
                )
            self.store.append_event(
                "shell_result",
                {
                    "step": step,
                    "command": action["command"],
                    **asdict(result),
                },
            )
            observation = (
                f"Command exit code: {result.return_code}\n"
                f"STDOUT:\n{_truncate(result.stdout, self.limits.max_observation_chars)}\n"
                f"STDERR:\n{_truncate(result.stderr, self.limits.max_observation_chars)}"
            )
            messages.append({"role": "user", "content": observation})
            if stop_after_action:
                status, summary = stop_after_action
                self.store.append_event(
                    status, {"step": step, "total_cost_usd": total_cost}
                )
                break

        outcome = AgentOutcome(
            status=status,
            steps=step,
            total_cost_usd=total_cost if cost_known else None,
            summary=summary,
            **token_totals,
        )
        self.store.write_summary(asdict(outcome))
        return outcome


def parse_action(content: str) -> dict[str, str]:
    candidate = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL)
    if fenced:
        candidate = fenced.group(1)
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        # Some models explain their next step and then emit one valid action.
        # Accept exactly one embedded JSON action while rejecting ambiguous
        # responses containing multiple action objects.
        decoder = json.JSONDecoder()
        embedded: list[object] = []
        for index, character in enumerate(candidate):
            if character != "{":
                continue
            try:
                parsed, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict) and parsed.get("action") in (
                "shell",
                "finish",
            ):
                embedded.append(parsed)
        if len(embedded) != 1:
            raise ValueError(
                "response must contain exactly one valid JSON action object"
            )
        value = embedded[0]
    if not isinstance(value, dict):
        raise ValueError("response must be a JSON object")
    action = value.get("action")
    if action == "shell":
        command = value.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError("shell action requires a non-empty command")
        return {"action": "shell", "command": command}
    if action == "finish":
        summary = value.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("finish action requires a non-empty summary")
        return {"action": "finish", "summary": summary}
    raise ValueError("action must be 'shell' or 'finish'")


def _truncate(value: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("max_observation_chars must be positive")
    if len(value) <= limit:
        return value
    head = limit // 2
    tail = limit - head
    omitted = len(value) - limit
    return f"{value[:head]}\n...[{omitted} characters omitted]...\n{value[-tail:]}"
