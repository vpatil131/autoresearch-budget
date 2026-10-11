import asyncio
import json
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory

from autoresearch_budget.advisor import AdvicePolicy, advice_packet, validate_advisor_routes
from autoresearch_budget.contracts import actor_tools, parse_actor_response, validate_guidance
from autoresearch_budget.models import OpenRouterClient, RequestOutcomeUnknown
from autoresearch_budget.store import RunStore


class AdvisorContractsTest(unittest.TestCase):
    def test_schedule_boundaries_and_deduplication(self):
        policy = AdvicePolicy()
        self.assertEqual([i for i in range(81) if policy.scheduled(i, 80, -1)], list(range(10, 80, 10)))
        self.assertFalse(policy.scheduled(10, 80, 10))
        self.assertFalse(AdvicePolicy(policy='actor_only').scheduled(10, 80, -1))
        for invalid in ({'every_steps': True}, {'max_calls': 0}, {'policy':'actor_requested'}, {'policy':'hybrid'}):
            with self.assertRaises(ValueError):
                AdvicePolicy.load(invalid)

    def test_packet_retains_oversized_recent_evidence_and_excludes_raw_payload(self):
        messages = [{'role':'system','content':'secret policy'}, {'role':'tool','content':'start'+'x'*10000+'end',
            'extra':{'observation_id':'observation-10','response':{'hidden':'never forward'}}}]
        packet = advice_packet(task='issue', messages=messages, diff='patch', question='review', actor_calls=10,
                              actor_limit=80, max_chars=1000, state={'phase':'validate'}, patch_id='patch-1')
        self.assertLessEqual(len(json.dumps(packet, ensure_ascii=False)), 1000)
        self.assertTrue(packet['context_truncated'])
        self.assertEqual(packet['recent_transcript'][0]['observation_id'], 'observation-10')
        self.assertIn('start', packet['recent_transcript'][0]['content'])
        self.assertNotIn('never forward', json.dumps(packet))
        self.assertNotIn('secret policy', json.dumps(packet))
        self.assertEqual(packet['actor_calls_remaining'], 70)

    def test_escape_heavy_packet_is_bounded(self):
        packet = advice_packet(task='\x00'*3000, messages=[{'role':'tool','content':'\x00'*5000}], diff='\x00'*3000,
            question='\x00'*500, actor_calls=10, actor_limit=80, max_chars=1000)
        self.assertLessEqual(len(json.dumps(packet, ensure_ascii=False)),1000)
        self.assertTrue(packet['recent_transcript'])

    def test_large_advice_history_remains_bounded_and_marks_truncation(self):
        packet=advice_packet(task='issue',messages=[{'role':'tool','content':'latest result'}],diff='patch',
            question='review',actor_calls=20,actor_limit=80,max_chars=1000,
            followthrough=[{'advice_id':'advice-1','actions':[{'command':'\x00'*10000}]}])
        self.assertLessEqual(len(json.dumps(packet,ensure_ascii=False)),1000)
        self.assertTrue(packet['context_truncated'])
        self.assertIn('advice_followthrough',packet)

    def test_route_validation_rejects_implicit_fallbacks_and_model_changes(self):
        provider={'only':['OpenAI'],'allow_fallbacks':False}
        fallback={'provider':{'only':['azure'],'allow_fallbacks':False},'max_tokens_parameter':'max_completion_tokens'}
        validate_advisor_routes({'provider':provider,'fallback':fallback})
        for invalid in ({**fallback,'model':'different'}, {**fallback,'service_tier':'flex'},
                        {**fallback,'provider':{'only':['azure'],'allow_fallbacks':True}}, {}, 'azure'):
            with self.subTest(fallback=invalid), self.assertRaises(ValueError):
                validate_advisor_routes({'provider':provider,'fallback':invalid})

    def test_actor_contract_rejects_prose_unknown_tools_and_invalid_state(self):
        with self.assertRaises(ValueError):
            parse_actor_response({'choices':[{'message':{'content':'{"command":"echo hi"}'}}]})
        for name, state in [('request_advice',{}),('execute',{'phase':'finished','uncertainty':'high','blocker':'none'})]:
            with self.assertRaises(ValueError):
                parse_actor_response({'choices':[{'message':{'tool_calls':[{'id':'x','type':'function','function':{
                    'name':name,'arguments':json.dumps({'command':'echo hi','state':state})}}]}}]})
        self.assertEqual({tool['function']['name'] for tool in actor_tools()}, {'execute','submit'})
        self.assertNotIn('schema_version',json.dumps(actor_tools()))

    def test_guidance_requires_the_agreed_five_fields(self):
        valid={'assessment':'evidence','gap':'gap','next_action':'act','validation':'test','assumptions':[]}
        self.assertEqual(validate_guidance(json.dumps(valid)),valid)
        for content in ('prose', '{}', json.dumps({**valid,'assumptions':'assume'}), json.dumps({**valid,'next_action':''})):
            with self.assertRaises(ValueError):
                validate_guidance(content)

    def test_recorded_retry_payload_usage_and_redaction(self):
        with TemporaryDirectory() as temp:
            store=RunStore(temp,run_id='demo',secrets=('secret-key',))
            responses=[(429,b'{"error":{"message":"slow"}}'),(200,json.dumps({'id':'r','model':'mock',
                'choices':[{'message':{'content':'ok'},'finish_reason':'stop'}],
                'usage':{'cost':0.02,'prompt_tokens':3}}).encode())]
            def transport(request, timeout):
                self.assertIn('Bearer secret-key',request.headers['Authorization'])
                return responses.pop(0)
            client=OpenRouterClient(api_key='secret-key',model='mock',allow_paid=True,transport=transport,
                                    retry_delays_sec=(0,),recorder=store.append_event)
            result=asyncio.run(client.complete([{'role':'user','content':'secret-key'}],request_id='demo:1',max_tokens=12))
            self.assertEqual(result.cost_usd,Decimal('0.02'))
            self.assertEqual(result.raw_response['id'],'r')
            events=[json.loads(line) for line in store.events_path.read_text().splitlines()]
            self.assertEqual([e['sequence'] for e in events],list(range(1,6)))
            self.assertEqual([e['kind'] for e in events],['model_request','model_response','model_retry','model_request','model_response'])
            self.assertNotIn('secret-key',store.events_path.read_text())
            self.assertEqual(events[3]['payload']['transport_attempt'],2)
            first=store.write_artifact('patch',{'content':'first'})
            second=store.write_artifact('patch',{'content':'second'})
            self.assertNotEqual(first,second)
            self.assertEqual(json.loads((Path(temp)/first).read_text())['content'],'first')

    def test_unknown_transport_outcome_is_preserved(self):
        events=[]
        def transport(*args):
            raise RequestOutcomeUnknown('timeout')
        client=OpenRouterClient(api_key='key',model='mock',allow_paid=True,transport=transport,
                                recorder=lambda kind,payload:events.append((kind,payload)))
        with self.assertRaises(RequestOutcomeUnknown):
            asyncio.run(client.complete([],request_id='x',max_tokens=12))
        self.assertEqual(events[-1][0],'model_transport_error')
        self.assertTrue(events[-1][1]['outcome_unknown'])


if __name__ == '__main__':
    unittest.main()
