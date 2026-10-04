import asyncio
import json
from decimal import Decimal
import unittest

from autoresearch_budget.models import OpenRouterClient


class OpenRouterClientTest(unittest.TestCase):
    def test_extracts_billed_cost_tokens_and_served_model(self):
        captured = {}

        def transport(request, timeout):
            captured["authorization"] = request.headers["Authorization"]
            captured["body"] = json.loads(request.data)
            captured["timeout"] = timeout
            response = {
                "id": "generation-1",
                "model": "qwen/qwen3.6-35b-a3b",
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"finish","summary":"ok"}'
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 20,
                    "cost": 0.012345,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 5},
                },
                "openrouter_metadata": {"provider_name": "example-provider"},
            }
            return 200, json.dumps(response).encode()

        client = OpenRouterClient(
            api_key="secret-value",
            model="qwen/qwen3.6-35b-a3b",
            provider={"allow_fallbacks": False},
            allow_paid=True,
            transport=transport,
        )
        completion = asyncio.run(
            client.complete(
                [{"role": "user", "content": "hello"}],
                request_id="request-1",
                max_tokens=1000,
            )
        )
        self.assertEqual(completion.cost_usd, Decimal("0.012345"))
        self.assertEqual(completion.cached_tokens, 40)
        self.assertEqual(completion.reasoning_tokens, 5)
        self.assertEqual(captured["body"]["max_completion_tokens"], 1000)
        self.assertEqual(captured["authorization"], "Bearer secret-value")

    def test_paid_client_requires_explicit_enablement(self):
        with self.assertRaises(ValueError):
            OpenRouterClient(api_key="secret", model="mock/model")

    def test_reasoning_only_response_preserves_cost_for_reconciliation(self):
        def transport(request, timeout):
            del request, timeout
            response = {
                "id": "reasoning-only",
                "model": "qwen/example",
                "choices": [
                    {
                        "message": {"content": None, "reasoning": "internal"},
                        "finish_reason": "length",
                    }
                ],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 128,
                    "cost": 0.004,
                    "completion_tokens_details": {"reasoning_tokens": 128},
                },
            }
            return 200, json.dumps(response).encode()

        client = OpenRouterClient(
            api_key="secret",
            model="qwen/example",
            allow_paid=True,
            transport=transport,
        )
        completion = asyncio.run(
            client.complete(
                [{"role": "user", "content": "probe"}],
                request_id="probe",
                max_tokens=128,
            )
        )
        self.assertEqual(completion.content, "")
        self.assertEqual(completion.cost_usd, Decimal("0.004"))
        self.assertEqual(completion.reasoning_tokens, 128)


if __name__ == "__main__":
    unittest.main()
