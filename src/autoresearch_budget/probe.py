"""Low-cost compatibility probes for configured model profiles."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from .agent import SYSTEM_PROMPT, parse_action
from .budget import BudgetLedger
from .models import CompletionBackend, OpenRouterClient
from .runner import ExperimentConfig, RunnerConfigError, _read_env_file
from .store import RunStore


@dataclass(frozen=True)
class ProbeResult:
    profile: str
    requested_model: str
    served_model: str
    response_id: str
    cost_usd: Decimal
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    reasoning_tokens: int
    finish_reason: str | None
    action_format_valid: bool
    action_format_error: str | None
    provider_metadata: dict[str, object]
    run_dir: str


async def run_probe(
    config: ExperimentConfig,
    *,
    profile_name: str,
    max_cost_usd: Decimal,
    max_tokens: int,
    backend: CompletionBackend | None = None,
) -> ProbeResult:
    if not config.paid_inference_enabled and backend is None:
        raise RunnerConfigError("paid inference is disabled in the experiment config")
    if profile_name not in config.profiles:
        raise RunnerConfigError(f"unknown model profile {profile_name!r}")
    if max_cost_usd <= 0:
        raise RunnerConfigError("probe max cost must be positive")
    if max_tokens < 1:
        raise RunnerConfigError("probe max tokens must be positive")

    profile = config.profiles[profile_name]
    if backend is None:
        environment = dict(os.environ)
        environment.update(_read_env_file(config.project_root / ".env"))
        api_key = environment.get("OPENROUTER_API_KEY") or environment.get(
            "OPENROUTER_KEY"
        )
        if not api_key:
            raise RunnerConfigError(
                "OPENROUTER_API_KEY or OPENROUTER_KEY is missing from the environment"
            )
        backend = OpenRouterClient(
            api_key=api_key,
            model=profile.model,
            reasoning=(
                {"effort": profile.reasoning_effort}
                if profile.reasoning_effort
                else None
            ),
            provider=profile.provider,
            allow_paid=True,
        )

    now = datetime.now(timezone.utc)
    run_id = f"probe-{profile_name}-{now:%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = config.project_root / "runs" / "probes" / run_id
    store = RunStore(run_dir)
    reservation_id = f"probe:{run_id}"
    ledger = BudgetLedger(config.budget_db, config.budget_cap_usd)
    ledger.reserve(
        reservation_id,
        run_id=run_id,
        model=profile.model,
        max_cost_usd=max_cost_usd,
    )
    store.write_metadata(
        {
            "run_id": run_id,
            "profile": profile_name,
            "model": profile.model,
            "max_cost_usd": max_cost_usd,
            "max_tokens": max_tokens,
            "reasoning_effort": profile.reasoning_effort,
            "provider": profile.provider,
        }
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "This is an adapter compatibility probe. Do not inspect or edit any "
                "files. Return the finish action with a short summary confirming the "
                "response format."
            ),
        },
    ]
    try:
        completion = await backend.complete(
            messages, request_id=run_id, max_tokens=max_tokens
        )
    except Exception as error:
        ledger.mark_uncertain(
            reservation_id,
            f"probe failed after dispatch: {type(error).__name__}: {error}",
        )
        store.append_event(
            "probe_error", {"type": type(error).__name__, "message": str(error)}
        )
        raise

    store.append_event(
        "model_response",
        {
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
    if completion.cost_usd is None:
        ledger.mark_uncertain(reservation_id, "probe response omitted usage.cost")
        raise RunnerConfigError(
            f"{profile_name} response omitted usage.cost; reservation remains held"
        )
    ledger.reconcile(reservation_id, completion.cost_usd)

    action_error: str | None = None
    try:
        action = parse_action(completion.content)
        if action["action"] != "finish":
            action_error = "probe returned a shell action instead of finish"
    except ValueError as error:
        action_error = str(error)

    result = ProbeResult(
        profile=profile_name,
        requested_model=completion.requested_model,
        served_model=completion.served_model,
        response_id=completion.response_id,
        cost_usd=completion.cost_usd,
        prompt_tokens=completion.prompt_tokens,
        completion_tokens=completion.completion_tokens,
        cached_tokens=completion.cached_tokens,
        reasoning_tokens=completion.reasoning_tokens,
        finish_reason=completion.finish_reason,
        action_format_valid=action_error is None,
        action_format_error=action_error,
        provider_metadata=completion.provider_metadata,
        run_dir=str(run_dir),
    )
    store.write_summary(asdict(result))
    return result

