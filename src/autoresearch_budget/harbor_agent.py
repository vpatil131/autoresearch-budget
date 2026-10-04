"""Harbor external-agent adapter for the project-owned minimal loop.

Harbor imports this module only during benchmark runs. Keeping the adapter thin
lets the core scaffold and budget logic be tested without installing Harbor.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import Any

from .agent import AgentLimits, ExecResult, MinimalShellAgent
from .budget import BudgetLedger
from .models import OpenRouterClient, RequestOutcomeUnknown, ScriptedModel
from .store import RunStore

try:
    from harbor.agents.base import BaseAgent
    from harbor.environments.base import BaseEnvironment
    from harbor.models.agent.context import AgentContext, ModelUsage
except ImportError as error:  # pragma: no cover - exercised only by Harbor
    raise ImportError(
        "harbor_agent must be imported by a Harbor environment; install harbor==0.23.0"
    ) from error


class _HarborSandbox:
    def __init__(self, environment: BaseEnvironment):
        self.environment = environment

    async def exec(self, command: str, *, timeout_sec: int) -> ExecResult:
        result = await self.environment.exec(command=command, timeout_sec=timeout_sec)
        return ExecResult(
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            return_code=result.return_code,
        )


class BudgetShellAgent(BaseAgent):
    """Host-side model loop; only shell commands cross into the task container."""

    @staticmethod
    def name() -> str:
        return "budget-shell"

    def version(self) -> str:
        return "0.1.0"

    def __init__(
        self,
        *args: Any,
        budget_db: str,
        budget_cap_usd: str,
        run_max_cost_usd: str,
        max_steps: int = 40,
        max_tokens_per_call: int = 4096,
        command_timeout_sec: int = 300,
        max_observation_chars: int = 20_000,
        reasoning_effort: str | None = None,
        provider: dict[str, Any] | None = None,
        allow_paid: bool = False,
        scripted_responses: list[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        if not self.model_name:
            raise ValueError("BudgetShellAgent requires --model")
        self.budget_db = Path(budget_db).expanduser().resolve()
        self.budget_cap_usd = Decimal(str(budget_cap_usd))
        self.run_max_cost_usd = Decimal(str(run_max_cost_usd))
        self.limits = AgentLimits(
            max_steps=int(max_steps),
            max_tokens_per_call=int(max_tokens_per_call),
            command_timeout_sec=int(command_timeout_sec),
            max_total_cost_usd=self.run_max_cost_usd,
            max_observation_chars=int(max_observation_chars),
        )
        self.reasoning = (
            {"effort": reasoning_effort} if reasoning_effort else None
        )
        self.provider = provider
        self.allow_paid = bool(allow_paid)
        self.scripted_responses = scripted_responses

    async def setup(self, environment: BaseEnvironment) -> None:
        del environment

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        run_id = str(self.context_id or self.session_id or self.logs_dir.name)
        reservation_id = f"harbor:{run_id}"
        ledger = BudgetLedger(self.budget_db, self.budget_cap_usd)
        ledger.reserve(
            reservation_id,
            run_id=run_id,
            model=self.model_name or "",
            max_cost_usd=self.run_max_cost_usd,
        )
        store = RunStore(self.logs_dir / "autoresearch")
        store.write_metadata(
            {
                "run_id": run_id,
                "model": self.model_name,
                "scaffold": self.name(),
                "scaffold_version": self.version(),
                "limits": self.limits,
                "reservation_id": reservation_id,
                "reserved_cost_usd": self.run_max_cost_usd,
                "network": self.scripted_responses is None,
            }
        )
        if self.scripted_responses is not None:
            model = ScriptedModel(
                self.scripted_responses, model=self.model_name or "mock/scripted"
            )
        else:
            api_key = self._get_env("OPENROUTER_API_KEY", "OPENROUTER_KEY")
            if not api_key:
                ledger.cancel(reservation_id, "missing API key before dispatch")
                raise ValueError("OPENROUTER_API_KEY or OPENROUTER_KEY is required")
            try:
                model = OpenRouterClient(
                    api_key=api_key,
                    model=self.model_name or "",
                    reasoning=self.reasoning,
                    provider=self.provider,
                    allow_paid=self.allow_paid,
                )
            except Exception:
                ledger.cancel(reservation_id, "model client setup failed before dispatch")
                raise
        scaffold = MinimalShellAgent(
            model=model,
            sandbox=_HarborSandbox(environment),
            store=store,
            run_id=run_id,
            limits=self.limits,
        )
        try:
            outcome = await scaffold.run(instruction)
        except RequestOutcomeUnknown as error:
            ledger.mark_uncertain(reservation_id, f"provider outcome unknown: {error}")
            raise
        except Exception:
            if self.scripted_responses is not None:
                ledger.cancel(reservation_id, "scripted run failed without network use")
            else:
                ledger.mark_uncertain(
                    reservation_id,
                    "run failed after reservation; reconcile against provider activity",
                )
            raise

        if outcome.total_cost_usd is None:
            ledger.mark_uncertain(
                reservation_id, "one or more responses omitted usage.cost"
            )
        else:
            ledger.reconcile(reservation_id, outcome.total_cost_usd)

        patch_result = await environment.exec(
            command="git diff --binary --no-ext-diff", timeout_sec=60
        )
        status_result = await environment.exec(
            command="git status --short", timeout_sec=60
        )
        (store.run_dir / "final.patch").write_text(
            patch_result.stdout or "", encoding="utf-8"
        )
        (store.run_dir / "final-status.txt").write_text(
            status_result.stdout or "", encoding="utf-8"
        )
        context.n_input_tokens = outcome.prompt_tokens
        context.n_cache_tokens = outcome.cached_tokens
        context.n_output_tokens = outcome.completion_tokens
        context.cost_usd = (
            float(outcome.total_cost_usd)
            if outcome.total_cost_usd is not None
            else None
        )
        context.model_usage = {
            self.model_name or "unknown": ModelUsage(
                n_input_tokens=outcome.prompt_tokens,
                n_cache_tokens=outcome.cached_tokens,
                n_output_tokens=outcome.completion_tokens,
                cost_usd=context.cost_usd,
            )
        }
        context.metadata = {
            "autoresearch_run_id": run_id,
            "agent_status": outcome.status,
            "reasoning_tokens": outcome.reasoning_tokens,
            "reservation_id": reservation_id,
        }
