"""Native phase 1 tools with recorded OpenRouter requests and responses.

Use mini-SWE's message/observation formatting, with the project's gateway for
observable transport attempts. Model and provider routes remain configurable.
"""
from __future__ import annotations

import asyncio
import os
import time

from minisweagent.exceptions import FormatError
from minisweagent.models.openrouter_model import OpenRouterModel

from .advisor import amount
from .contracts import actor_tools, parse_actor_response
from .models import OpenRouterClient


class StructuredActorModel(OpenRouterModel):
    def __init__(self, *, backend=None, **kwargs):
        super().__init__(**kwargs)
        self.backend = backend
        self.store = None
        self.actor_call = 0
        self.pending_advice = None
        self.observation_ids: set[str] = set()

    def bind(self, store):
        self.store = store

    def _record(self, kind, payload):
        if self.store:
            self.store.append_event(kind, {"model_role": "actor", "actor_call": self.actor_call, **payload})

    def _client(self):
        if self.backend is None:
            options = dict(self.config.model_kwargs)
            extra = options.pop("extra_body", {})
            provider = extra.pop("provider", None)
            effort = options.pop("reasoning_effort", None)
            reasoning = extra.pop("reasoning", None) or options.pop("reasoning", None)
            max_tokens = options.pop("max_tokens", 4096)
            timeout = options.pop("timeout", 180)
            options.pop("drop_params", None)
            options.pop("extra_headers", None)  # Session correlation is in request metadata.
            options.pop("api_key", None)
            self.backend = OpenRouterClient(
                api_key=os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MSWEA_API_KEY", ""),
                model=self.config.model_name.removeprefix("openrouter/"),
                provider=provider, reasoning=reasoning or ({"effort": effort} if effort else None),
                service_tier=options.pop("service_tier", None), timeout_sec=timeout,
                tools=actor_tools(), body_options={**extra, **options, "tool_choice": "required", "parallel_tool_calls": False},
                recorder=self._record, allow_paid=True,
            )
            self.max_tokens = max_tokens
        return self.backend

    def query(self, messages: list[dict], **kwargs) -> dict:
        if kwargs:
            raise ValueError("Per-call actor overrides are not supported in phase 1")
        prepared = self._prepare_messages_for_api(messages)
        request_id = f"{self.store.run_id if self.store else 'actor'}:actor:{self.actor_call}"
        self._record("actor_input", {"request_id": request_id, "messages": prepared, "tools": actor_tools()})
        started = time.monotonic()
        completion = asyncio.run(self._client().complete(prepared, request_id=request_id,
                                  max_tokens=getattr(self, "max_tokens", self.config.model_kwargs.get("max_tokens", 4096))))
        response = completion.raw_response
        cost = amount(completion.cost_usd)
        extra = {"cost": float(cost or 0), "response": response, "timestamp": time.time(), "request_id": request_id}
        self._record("actor_completion", {"request_id": request_id, "response": response,
            "duration_seconds": time.monotonic() - started, "cost_usd": cost, "usage": completion.raw_usage})
        try:
            action = parse_actor_response(response)
            decision = action.get("advice_decision")
            if self.pending_advice:
                if not decision or decision["advice_id"] != self.pending_advice:
                    raise ValueError(f"advice_decision for {self.pending_advice} is required")
            elif decision:
                raise ValueError("No advisor guidance is awaiting a decision")
            if action["name"] == "submit" and set(action["validation_refs"]) - self.observation_ids:
                raise ValueError("validation_refs contains unknown observation IDs")
        except (ValueError, TypeError, KeyError) as error:
            self._record("actor_schema_error", {"request_id": request_id, "error": str(error)})
            raise FormatError({"role": "user", "content": f"Tool schema error: {error}. Use exactly one execute or submit tool with valid structured state.",
                               "extra": {**extra, "interrupt_type": "FormatError"}}) from error
        message = dict(response["choices"][0]["message"])
        message["extra"] = {**extra, "actions": [action]}
        return message
