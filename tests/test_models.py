import asyncio
import json
from decimal import Decimal
import unittest

from autoresearch_budget.models import FallbackClient, ModelError, OpenRouterClient, RequestOutcomeUnknown


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
                "service_tier": "flex",
            }
            return 200, json.dumps(response).encode()

        client = OpenRouterClient(
            api_key="secret-value",
            model="qwen/qwen3.6-35b-a3b",
            service_tier="flex",
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
        self.assertEqual(captured["body"]["max_tokens"], 1000)
        self.assertEqual(captured["body"]["service_tier"], "flex")
        self.assertEqual(completion.provider_metadata["service_tier"], "flex")
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

    def test_retries_explicit_429_before_returning_completion(self):
        responses = [
            (
                429,
                json.dumps(
                    {"error": {"message": "Provider returned error"}}
                ).encode(),
            ),
            (
                200,
                json.dumps(
                    {
                        "id": "generation-2",
                        "model": "qwen/example",
                        "choices": [
                            {
                                "finish_reason": "stop",
                                "message": {
                                    "content": '{"action":"finish","summary":"ok"}'
                                },
                            }
                        ],
                        "usage": {"cost": 0.001},
                    }
                ).encode(),
            ),
        ]

        def transport(request, timeout):
            del request, timeout
            return responses.pop(0)

        client = OpenRouterClient(
            api_key="test",
            model="qwen/example",
            allow_paid=True,
            transport=transport,
            retry_delays_sec=(0,),
        )
        completion = asyncio.run(
            client.complete([], request_id="retry-test", max_tokens=100)
        )
        self.assertEqual(completion.response_id, "generation-2")
        self.assertEqual(responses, [])


class AdvisorFallbackTest(unittest.TestCase):
    def client(self, primary_result, secondary_result=None):
        from autoresearch_budget.advisor import Advisor
        self.requests=[]; self.events=[]
        advisor=Advisor.from_profile({'model':'openai/gpt-6.1-sol','reasoning_effort':'medium',
            'service_tier':'flex','provider':{'only':['OpenAI'],'allow_fallbacks':False},
            'fallback':{'provider':{'only':['azure'],'allow_fallbacks':False,'require_parameters':True},
                        'max_tokens_parameter':'max_completion_tokens'}},'test-key',4096,
            lambda kind,payload:self.events.append((kind,payload)))
        def transport(result):
            def call(request,timeout):
                self.requests.append(json.loads(request.data))
                if isinstance(result,Exception):raise result
                return result[0],json.dumps(result[1]).encode()
            return call
        advisor.backend.primary.transport=transport(primary_result)
        advisor.backend.fallback.transport=transport(secondary_result or self.success())
        return advisor.backend

    def success(self):
        return 200,{'id':'ok','model':'openai/gpt-6.1-sol','provider':'Azure',
            'choices':[{'message':{'content':'guidance'},'finish_reason':'stop'}],
            'usage':{'cost':0.02,'prompt_tokens':20,'completion_tokens':10}}

    def complete(self,client):
        return asyncio.run(client.complete([{'role':'user','content':'identical context'}],request_id='logical:advice-1',max_tokens=4096))

    def test_primary_success_never_calls_azure(self):
        result=self.complete(self.client(self.success()))
        self.assertEqual(len(self.requests),1)
        self.assertEqual(len(result.attempts),1)
        self.assertFalse(any(k=='model_fallback' for k,_ in self.events))

    def test_observed_embedded_502_falls_back_and_adapts_azure_request(self):
        error={'error':{'message':'Flex processing is temporarily unavailable.','code':502,
                       'metadata':{'error_type':'provider_unavailable'}},
               'usage':{'cost':0.003,'prompt_tokens':8,'completion_tokens':4}}
        result=self.complete(self.client((200,error)))
        first,second=self.requests
        self.assertEqual(first['provider']['only'],['OpenAI'])
        self.assertEqual(first['service_tier'],'flex')
        self.assertEqual(second['provider']['only'],['azure'])
        self.assertNotIn('service_tier',second)
        self.assertNotIn('max_tokens',second)
        self.assertEqual(second['max_completion_tokens'],first['max_tokens'])
        for key in ('messages','model','reasoning','response_format'):
            self.assertEqual(first[key],second[key])
        self.assertEqual([a['cost_usd'] for a in result.attempts],[0.003,0.02])
        self.assertEqual([a['route_attempt'] for a in result.attempts],[1,2])
        self.assertNotEqual(first['metadata'],second['metadata'])
        self.assertEqual(sum(k=='model_request' for k,_ in self.events),2)

    def test_availability_errors_and_unknown_timeout_have_bounded_fallback(self):
        for failure in ((429,{'error':{'message':'rate limit'}}),(503,{'error':{'message':'unavailable'}}),
                        RequestOutcomeUnknown('timeout')):
            with self.subTest(failure=failure):
                result=self.complete(self.client(failure))
                self.assertEqual(len(self.requests),2)
                self.assertIsNone(result.attempts[0]['cost_usd'])
                self.assertEqual(result.attempts[0]['outcome_unknown'],isinstance(failure,RequestOutcomeUnknown))

    def test_non_json_server_failure_can_fallback(self):
        client=self.client(self.success())
        client.primary.transport=lambda request,timeout:(502,b'Bad Gateway')
        result=self.complete(client)
        self.assertEqual(len(result.attempts),2)

    def test_auth_bad_request_and_malformed_success_do_not_fallback(self):
        for status in (400,401,403,404,422):
            with self.subTest(status=status):
                with self.assertRaises(ModelError):self.complete(self.client((status,{'error':{'message':'invalid'}})))
                self.assertEqual(len(self.requests),1)
        with self.assertRaises(ModelError):self.complete(self.client((200,{'choices':[]})))
        self.assertEqual(len(self.requests),1)

    def test_both_routes_fail_once_and_preserve_all_attempt_usage(self):
        client=self.client((200,{'error':{'code':502}}),(429,{'error':{'message':'rate limited'},'usage':{'cost':0.01}}))
        with self.assertRaises(ModelError) as raised:self.complete(client)
        self.assertEqual(len(self.requests),2)
        self.assertEqual(len(raised.exception.attempts),2)
        self.assertEqual(raised.exception.attempts[1]['cost_usd'],0.01)

    def test_fallback_cannot_change_model_or_hide_retries(self):
        first=OpenRouterClient(api_key='x',model='a',allow_paid=True,retry_delays_sec=())
        other=OpenRouterClient(api_key='x',model='b',allow_paid=True,retry_delays_sec=())
        with self.assertRaises(ValueError):FallbackClient(first,other)
        with self.assertRaises(ValueError):FallbackClient(first,OpenRouterClient(api_key='x',model='a',allow_paid=True))


if __name__ == "__main__":
    unittest.main()
