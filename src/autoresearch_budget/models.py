"""Model gateway contracts and a non-streaming OpenRouter adapter."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Callable, Protocol


class ModelError(RuntimeError):
    pass


class RequestOutcomeUnknown(ModelError):
    """The request may have reached the provider, so its cost is unknown."""


@dataclass(frozen=True)
class Completion:
    content: str
    response_id: str
    requested_model: str
    served_model: str
    cost_usd: Decimal | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    finish_reason: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    raw_usage: dict[str, Any] = field(default_factory=dict)


class CompletionBackend(Protocol):
    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        request_id: str,
        max_tokens: int,
    ) -> Completion: ...


Transport = Callable[[urllib.request.Request, float], tuple[int, bytes]]


def _default_transport(
    request: urllib.request.Request, timeout_sec: float
) -> tuple[int, bytes]:
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        raise RequestOutcomeUnknown(str(error)) from error


class OpenRouterClient:
    """Small adapter with no hidden retries or model fallbacks."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        reasoning: dict[str, Any] | None = None,
        service_tier: str | None = None,
        provider: dict[str, Any] | None = None,
        allow_paid: bool = False,
        endpoint: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout_sec: float = 180.0,
        transport: Transport = _default_transport,
        retry_delays_sec: tuple[float, ...] = (2.0, 4.0, 8.0),
    ):
        if not allow_paid:
            raise ValueError("paid inference is disabled; pass allow_paid=True explicitly")
        if not api_key:
            raise ValueError("OpenRouter API key is empty")
        self.api_key = api_key
        self.model = model
        self.reasoning = reasoning
        self.service_tier = service_tier
        self.provider = provider
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec
        self.transport = transport
        self.retry_delays_sec = retry_delays_sec

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        request_id: str,
        max_tokens: int,
    ) -> Completion:
        import asyncio

        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "stream": False,
            "metadata": {"experiment_request_id": request_id},
        }
        if self.reasoning:
            body["reasoning"] = self.reasoning
        if self.service_tier:
            body["service_tier"] = self.service_tier
        if self.provider:
            body["provider"] = self.provider
        request = urllib.request.Request(
            self.endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "autoresearch-budget/0.1.0",
                "X-OpenRouter-Metadata": "enabled",
            },
            method="POST",
        )
        for attempt in range(len(self.retry_delays_sec) + 1):
            status, raw = await asyncio.to_thread(
                self.transport, request, self.timeout_sec
            )
            try:
                response = json.loads(raw)
            except json.JSONDecodeError as error:
                raise ModelError(
                    f"OpenRouter returned non-JSON HTTP {status}"
                ) from error
            if status != 429 or attempt == len(self.retry_delays_sec):
                break
            await asyncio.sleep(self.retry_delays_sec[attempt])
        if status < 200 or status >= 300:
            raise ModelError(f"OpenRouter HTTP {status}: {_error_message(response)}")
        if response.get("error"):
            raise ModelError(f"OpenRouter error: {_error_message(response)}")
        choices = response.get("choices") or []
        if not choices:
            raise ModelError("OpenRouter response contained no choices")
        choice = choices[0]
        if choice.get("error") or choice.get("finish_reason") == "error":
            raise ModelError(f"OpenRouter generation error: {_error_message(choice)}")
        message = choice.get("message") or {}
        content = message.get("content")
        # Reasoning models can consume their entire output allowance before
        # emitting final text. Preserve the usage and billed cost so the caller
        # can reconcile the reservation, then treat the empty final as a format
        # failure at the agent/probe layer.
        if content is None:
            content = ""
        elif not isinstance(content, str):
            raise ModelError("OpenRouter response contained no text content")
        usage = response.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        cost = usage.get("cost")
        return Completion(
            content=content,
            response_id=str(response.get("id", "")),
            requested_model=self.model,
            served_model=str(response.get("model", self.model)),
            cost_usd=Decimal(str(cost)) if cost is not None else None,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cached_tokens=int(prompt_details.get("cached_tokens") or 0),
            reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            finish_reason=choice.get("finish_reason"),
            provider_metadata={
                **(response.get("openrouter_metadata") or {}),
                "service_tier": response.get("service_tier"),
            },
            raw_usage=usage,
        )


def _error_message(value: dict[str, Any]) -> str:
    error = value.get("error", value)
    if isinstance(error, dict):
        return str(error.get("message") or error.get("code") or error)
    return str(error)


class ScriptedModel:
    """Deterministic zero-network backend for scaffold tests."""

    def __init__(self, responses: list[str], *, model: str = "mock/scripted"):
        self.responses = list(responses)
        self.model = model
        self.requests: list[list[dict[str, str]]] = []

    async def complete(
        self,
        messages: list[dict[str, str]],
        *,
        request_id: str,
        max_tokens: int,
    ) -> Completion:
        del max_tokens
        if not self.responses:
            raise ModelError("scripted model exhausted")
        self.requests.append([dict(message) for message in messages])
        return Completion(
            content=self.responses.pop(0),
            response_id=request_id,
            requested_model=self.model,
            served_model=self.model,
            cost_usd=Decimal("0"),
        )
