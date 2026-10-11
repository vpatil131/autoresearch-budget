"""Offline integration tests against mini-SWE 2.4.6; no provider calls."""
import importlib.util
import json
import subprocess
import unittest
from decimal import Decimal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import AsyncMock, Mock, patch

from autoresearch_budget.models import Completion, OpenRouterClient, ModelError, RequestOutcomeUnknown

HAS_MINI = importlib.util.find_spec('minisweagent') is not None
if HAS_MINI:
    from autoresearch_budget.actor_model import StructuredActorModel
    from autoresearch_budget.advisory_agent import AdvisoryAgent
    from autoresearch_budget.advisor import Advisor, AdvisorUnavailable, SCHEDULED_QUESTION
    from autoresearch_budget.poc_environment import PocEnvironment
    from minisweagent.config import get_config_from_spec
    from minisweagent.models import get_model
    from minisweagent.environments import get_environment
    from minisweagent.agents import get_agent
    from minisweagent.utils.serialize import recursive_merge

STATE={'phase':'inspect','uncertainty':'medium','blocker':'none'}
GUIDANCE={'assessment':'Read the relevant caller','gap':'No regression check yet',
          'next_action':'Add a targeted regression check','validation':'Run it','assumptions':['Caller matters']}


def adopt(advice_id):
    return {'advice_id':advice_id,'decision':'adopt','reason':'Test the proposed hypothesis',
            'next_action':'Run a targeted public check','validation_plan':'Check its exit code and output'}


def progress(advice_id, ref, *, status='validated'):
    return {'advice_id':advice_id,'status':status,'reason':'The linked public check provides evidence',
            'evidence_refs':[ref] if ref else []}


def response(index, *, name='execute', args=None):
    args=args or {'command':'true','state':STATE}
    return {'id':f'model-{index}','model':'mock/actor','choices':[{'message':{'role':'assistant',
        'content':'brief reasoning','tool_calls':[{'id':f'tool-{index}','type':'function','function':{
            'name':name,'arguments':json.dumps(args)}}]},'finish_reason':'tool_calls'}],
        'usage':{'cost':0.001,'prompt_tokens':10,'completion_tokens':5}}


@unittest.skipUnless(HAS_MINI, 'Install mini-SWE-agent 2.4.6 for runtime integration tests')
class AdvisoryRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.repo=self.root/'repo'; self.repo.mkdir()
        self.git('init','-q')
        self.git('config','user.email','offline@example.test')
        self.git('config','user.name','Offline test')
        (self.repo/'tracked.txt').write_text('before\n')
        self.git('add','tracked.txt'); self.git('commit','-qm','initial')

    def git(self,*args):
        return subprocess.run(['git',*args],cwd=self.repo,check=True,capture_output=True,text=True)

    def agent(self,responses,*,limit=80,k=10,advisor_content=None,advisor=None):
        self.requests=[]
        queue=list(responses)
        def transport(request, timeout):
            body=json.loads(request.data); self.requests.append(body)
            if not queue:
                raise AssertionError('Unexpected model call')
            return 200,json.dumps(queue.pop(0)).encode()
        actor=StructuredActorModel(model_name='openrouter/mock/actor',backend=OpenRouterClient(
            api_key='offline-test-key',model='mock/actor',allow_paid=True,transport=transport),
            observation_template='{{ output.output }}')
        env=PocEnvironment(cwd=str(self.repo),env={'OPENROUTER_API_KEY':'','MSWEA_API_KEY':''})
        if advisor is None:
            def advice_transport(request,timeout):
                return 200,json.dumps({'id':'advice-model','model':'mock/advisor','choices':[{'message':{
                    'content':advisor_content if advisor_content is not None else json.dumps(GUIDANCE)},'finish_reason':'stop'}],
                    'usage':{'cost':0.02,'prompt_tokens':20,'completion_tokens':10}}).encode()
            advisor=Advisor(OpenRouterClient(api_key='offline-test-key',model='mock/advisor',allow_paid=True,
                                           transport=advice_transport),128)
        agent=AdvisoryAgent(actor,env,advice={'every_steps':k,'run_id':'offline-task'},advisor=advisor,
            cost_limit=0,step_limit=limit,system_template='ignored',instance_template='ignored',
            output_path=self.root/'agent'/'mini-swe-agent.trajectory.json',max_observation_chars=200)
        actor.backend.tools=__import__('autoresearch_budget.contracts',fromlist=['actor_tools']).actor_tools()
        actor.backend.recorder=actor._record
        if hasattr(advisor,'backend'):
            advisor.backend.recorder=agent._advisor_record
            if hasattr(advisor.backend,'primary'):
                advisor.backend.primary.recorder=advisor.backend.fallback.recorder=agent._advisor_record
        return agent

    def events(self, agent):
        return [json.loads(line) for line in agent.advice_store.events_path.read_text().splitlines()]

    def test_full_80_call_schedule_and_replay_records(self):
        responses=[]
        for index in range(1,81):
            args={'command':'true','state':STATE}
            if index%10==1 and index>1:
                args['advice_decision']=adopt(f'advice-{index//10}')
            if index%10==2 and index>10:
                args['advice_progress']=[progress(f'advice-{index//10}',f'observation-{index-1}')]
            responses.append(response(index,args=args))
        agent=self.agent(responses)
        outcome=agent.run('Fix the public issue')
        self.assertEqual(outcome['exit_status'],'LimitsExceeded')
        self.assertEqual(agent.n_calls,80)
        events=self.events(agent)
        consultations=[e for e in events if e['kind']=='advice_requested']
        self.assertEqual([e['payload']['actor_call'] for e in consultations],list(range(10,80,10)))
        self.assertTrue(all(e['payload']['packet']['question']==SCHEDULED_QUESTION for e in consultations))
        self.assertEqual(len([e for e in events if e['kind']=='advice_decision']),7)
        self.assertEqual(len([e for e in events if e['kind']=='model_request']),87)
        self.assertEqual(len([e for e in events if e['kind']=='model_response']),87)
        self.assertEqual([e['sequence'] for e in events],list(range(1,len(events)+1)))
        self.assertTrue(all(e['run_id']=='offline-task' for e in events))
        self.assertIn('actor_state',consultations[0]['payload']['packet'])
        self.assertEqual(json.loads((self.root/'agent'/'advice'/'summary.json').read_text())['total_cost_usd'],'0.220')
        self.assertNotIn('advice_decision',self.requests[0]['tools'][0]['function']['parameters']['properties'])
        pending=self.requests[10]['tools'][0]['function']['parameters']
        self.assertIn('advice_decision',pending['required'])
        self.assertEqual(pending['properties']['advice_decision']['properties']['advice_id']['enum'],['advice-1'])
        self.assertIn('advice_progress',self.requests[11]['tools'][0]['function']['parameters']['required'])
        self.assertTrue(all(r['status']=='validated' for r in agent.advice_records.values()))

    def test_submission_at_checkpoint_skips_advice_and_preserves_candidate(self):
        responses=[response(i) for i in range(1,10)]
        responses[0]=response(1,args={'command':"printf 'after\\n' > tracked.txt; printf 'new\\n' > new.txt",'state':{**STATE,'hypothesis':'Fix the consumer'}})
        responses.append(response(10,name='submit',args={'summary':'Candidate ready','validation_refs':['observation-9'],'state':STATE}))
        agent=self.agent(responses)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,0)
        self.assertEqual(agent.state['hypothesis'],'Fix the consumer')
        patches=[e for e in self.events(agent) if e['kind']=='patch_snapshot']
        final=json.loads((agent.advice_store.run_dir/patches[-1]['payload']['artifact']).read_text())['patch']
        self.assertIn('+after',final); self.assertIn('+new',final)
        self.assertEqual(patches[-1]['payload']['reason'],'final')

    def test_schema_error_counts_toward_checkpoint_and_requires_decision(self):
        bad=response(1,args={'command':'touch should-not-exist','state':{**STATE,'phase':'invalid'}})
        missing_decision=response(3,args={'command':'touch also-not-created','state':STATE})
        good=response(4,name='submit',args={'summary':'done','validation_refs':['observation-2'],'state':STATE,
            'advice_decision':{'advice_id':'advice-1','decision':'reject','reason':'The linked observation already checks the caller',
                               'evidence_refs':['observation-2']}})
        agent=self.agent([bad,response(2),missing_decision,good],k=2)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.n_calls,4); self.assertEqual(agent.advisor_calls,1)
        self.assertFalse((self.repo/'should-not-exist').exists())
        self.assertFalse((self.repo/'also-not-created').exists())
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['actor']['responses_with_cost'],4)
        self.assertEqual(len([e for e in self.events(agent) if e['kind']=='actor_schema_error']),2)

    def test_malformed_advice_preserves_usage_and_fails_attempt(self):
        end=response(3,name='submit',args={'summary':'done','validation_refs':[],'state':STATE})
        agent=self.agent([response(1),response(2),end],k=2,advisor_content='not JSON')
        with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
        self.assertIsNone(agent.pending_advice)
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advisor']['failures'],1)
        self.assertEqual(summary['advisor']['cost_usd'],'0.02')
        self.assertTrue(any(e['kind']=='advice_schema_error' for e in self.events(agent)))
        self.assertEqual(summary['stop_reason'],'AdvisorUnavailable')
        self.assertEqual(summary['actor']['api_calls'],2)

    def test_advisor_failure_keeps_unknown_cost_distinct(self):
        advisor=Mock(); advisor.consult.side_effect=ModelError('Unavailable')
        agent=self.agent([response(1),response(2),response(3,name='submit',args={'summary':'done','validation_refs':[],'state':STATE})],k=2,advisor=advisor)
        with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertIsNone(summary['advisor']['cost_usd']); self.assertIsNone(summary['total_cost_usd'])
        self.assertEqual(summary['advisor']['observed_cost_usd'],'0')

    def test_advisor_generation_error_retains_billed_usage(self):
        advisor=Mock(); advisor.consult.side_effect=ModelError('generation failed',response={
            'usage':{'cost':0.03,'prompt_tokens':8,'completion_tokens':4}})
        agent=self.agent([response(1),response(2),response(3,name='submit',args={
            'summary':'done','validation_refs':[],'state':STATE})],k=2,advisor=advisor)
        with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advisor']['cost_usd'],'0.03')
        self.assertEqual(summary['advisor']['completion_tokens'],4)
        self.assertEqual(summary['actor']['cost_usd'],'0.002')

    def test_actor_generation_error_records_returned_usage_and_final_candidate(self):
        agent=self.agent([{'id':'bad','choices':[{'message':{'content':''},'finish_reason':'error',
            'error':{'message':'failed'}}],'usage':{'cost':0.04,'prompt_tokens':8}}])
        with self.assertRaises(ModelError):
            agent.run('Issue')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['actor']['cost_usd'],'0.04')
        self.assertEqual(summary['stop_reason'],'ModelError')
        self.assertTrue(any(e['kind']=='patch_snapshot' and e['payload']['reason']=='final' for e in self.events(agent)))

    def test_adopt_executes_a_patch_and_requires_evidence_before_submission(self):
        args={'command':"printf 'after\\n' > tracked.txt; test \"$(cat tracked.txt)\" = after",'state':STATE,
              'advice_decision':adopt('advice-1')}
        premature=response(12,name='submit',args={'summary':'pretend done','validation_refs':[], 'state':STATE})
        finish=response(13,name='submit',args={'summary':'fixed','validation_refs':['observation-11'],'state':STATE,
            'advice_progress':[progress('advice-1','observation-11')]})
        agent=self.agent([*[response(i) for i in range(1,11)],response(11,args=args),premature,finish])
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        record=agent.advice_records['advice-1']
        self.assertEqual(record['status'],'validated')
        self.assertEqual(record['actions'][0]['observation_id'],'observation-11')
        self.assertEqual((self.repo/'tracked.txt').read_text(),'after\n')
        events=self.events(agent)
        self.assertTrue(any(e['kind']=='actor_schema_error' and 'advice_progress' in e['payload']['error'] for e in events))
        self.assertTrue(any(e['kind']=='advice_status' and e['payload']['status']=='in_progress' for e in events))
        self.assertIn('+after',agent.patch_text)

    def test_immediate_submission_after_adopt_and_stale_decisions_are_rejected(self):
        reject={'advice_id':'advice-1','decision':'reject','reason':'The observation rules out the assumption',
                'evidence_refs':['observation-2']}
        # K=4 avoids a second checkpoint while repairing the immediate submission.
        agent=self.agent([response(1),response(2),response(3),response(4),
            response(5,name='submit',args={'summary':'done','validation_refs':[], 'state':STATE,'advice_decision':adopt('advice-1')}),
            response(6,args={'command':'true','state':STATE,'advice_decision':{**reject,'evidence_refs':['observation-4']}}),
            response(7,args={'command':'touch stale-should-not-execute','state':STATE,'advice_decision':{**reject,'evidence_refs':['observation-4']}}),
            response(8,name='submit',args={'summary':'done','validation_refs':['observation-4'],'state':STATE})],k=4)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertFalse((self.repo/'stale-should-not-execute').exists())
        errors=[e['payload']['error'] for e in self.events(agent) if e['kind']=='actor_schema_error']
        self.assertTrue(any('executing the next action' in e for e in errors))
        self.assertTrue(any('No advisor guidance' in e for e in errors))

    def test_failed_or_unrelated_evidence_cannot_validate_advice(self):
        adopt_action=response(3,args={'command':'false','state':STATE,'advice_decision':adopt('advice-1')})
        wrong=response(4,name='submit',args={'summary':'done','validation_refs':[], 'state':STATE,
            'advice_progress':[progress('advice-1','observation-2')]})
        failed=response(5,name='submit',args={'summary':'done','validation_refs':[], 'state':STATE,
            'advice_decision':{'advice_id':'advice-2','decision':'reject','reason':'Existing failed check blocks this action','evidence_refs':['observation-3']},
            'advice_progress':[progress('advice-1','observation-3')]})
        blocked=response(6,name='submit',args={'summary':'blocked','validation_refs':['observation-3'], 'state':STATE,
            'advice_decision':{'advice_id':'advice-2','decision':'reject','reason':'Existing failed check blocks this action','evidence_refs':['observation-3']},
            'advice_progress':[progress('advice-1','observation-3',status='blocked')]})
        agent=self.agent([response(1),response(2),adopt_action,wrong,failed,blocked],k=2)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        errors=[e['payload']['error'] for e in self.events(agent) if e['kind']=='actor_schema_error']
        self.assertTrue(any('executed under that advice' in e for e in errors))
        self.assertTrue(any('successful command evidence' in e for e in errors))
        self.assertEqual(agent.advice_records['advice-1']['status'],'blocked')

    def test_open_advice_survives_next_checkpoint_and_is_reviewed(self):
        packets=[]
        def advice_transport(request,timeout):
            packets.append(json.loads(json.loads(request.data)['messages'][1]['content']))
            return 200,json.dumps({'id':'advice','choices':[{'message':{'content':json.dumps(GUIDANCE)}}],
                                   'usage':{'cost':0.02}}).encode()
        advisor=Advisor(OpenRouterClient(api_key='offline',model='mock/advisor',allow_paid=True,transport=advice_transport),128)
        agent=self.agent([response(1),response(2),
            response(3,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            response(4,args={'command':'true','state':STATE,'advice_progress':[progress('advice-1',None,status='in_progress')]}),
            response(5,args={'command':'true','state':STATE,'advice_decision':adopt('advice-2'),
                            'advice_progress':[progress('advice-1',None,status='in_progress')]}),
            response(6,name='submit',args={'summary':'done','validation_refs':['observation-5'],'state':STATE,
                'advice_progress':[progress('advice-1','observation-5'),progress('advice-2','observation-5')]})],k=2,advisor=advisor)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        history=json.loads(packets[1]['advice_followthrough'])
        self.assertEqual(history[0]['status'],'in_progress')
        self.assertEqual([a['actor_call'] for a in history[0]['actions']],[3,4])
        self.assertTrue(all(r['status']=='validated' for r in agent.advice_records.values()))

    def test_actor_limit_retains_unresolved_advice_without_extra_model_calls(self):
        agent=self.agent([*[response(i) for i in range(1,11)],
            response(11,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')})],limit=11)
        self.assertEqual(agent.run('Issue')['exit_status'],'LimitsExceeded')
        self.assertEqual(agent.n_calls,11)
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advice_followthrough']['advice-1']['status'],'in_progress')
        self.assertEqual(summary['advisor']['logical_calls'],1)

    def routed_advisor(self, *, fail_both=False, timeout=False):
        from autoresearch_budget.advisor import Advisor
        config=json.loads(Path('config/experiment.phase1.v2.json').read_text())
        advisor=Advisor.from_profile(config['profiles']['frontier'],'offline-secret',128)
        def primary(request,delay):
            if timeout:raise RequestOutcomeUnknown('timeout')
            return 200,json.dumps({'error':{'code':502,'message':'Flex processing is temporarily unavailable.',
                'metadata':{'error_type':'provider_unavailable'}},'usage':{'cost':0.003,'prompt_tokens':8,'completion_tokens':4}}).encode()
        def azure(request,delay):
            if fail_both:return 503,json.dumps({'error':{'message':'Azure unavailable'}}).encode()
            return 200,json.dumps({'id':'azure-response','model':config['profiles']['frontier']['model'],
                'choices':[{'message':{'content':json.dumps(GUIDANCE)}}],
                'usage':{'cost':0.02,'prompt_tokens':20,'completion_tokens':10}}).encode()
        advisor.backend.primary.transport=primary
        advisor.backend.fallback.transport=azure
        return advisor

    def test_k10_fallback_preserves_logical_schedule_usage_and_demo_records(self):
        advisor=self.routed_advisor()
        agent=self.agent([*[response(i) for i in range(1,11)],
            response(11,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            response(12,name='submit',args={'summary':'done','validation_refs':['observation-11'],'state':STATE,
                'advice_progress':[progress('advice-1','observation-11')]})],advisor=advisor)
        advisor.backend.recorder=agent._advisor_record
        advisor.backend.primary.recorder=advisor.backend.fallback.recorder=agent._advisor_record
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advisor']['logical_calls'],1)
        self.assertEqual(summary['advisor']['physical_api_calls'],2)
        self.assertEqual(summary['advisor']['cost_usd'],'0.023')
        self.assertEqual(summary['advisor']['prompt_tokens'],28)
        self.assertEqual(summary['advisor']['completion_tokens'],14)
        self.assertEqual(summary['total_cost_usd'],'0.035')
        requests=[e for e in self.events(agent) if e['kind']=='model_request' and e['payload']['model_role']=='advisor']
        self.assertEqual(len(requests),2)
        self.assertTrue(all(e['payload']['actor_call']==10 for e in requests))
        self.assertNotIn('offline-secret',agent.advice_store.events_path.read_text())

    def test_timeout_fallback_keeps_primary_cost_unknown(self):
        advisor=self.routed_advisor(timeout=True)
        agent=self.agent([response(1),response(2),response(3,name='submit',args={'summary':'done','validation_refs':[], 'state':STATE,
            'advice_decision':{'advice_id':'advice-1','decision':'reject','reason':'Observation rules out the proposal','evidence_refs':['observation-2']}})],k=2,advisor=advisor)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advisor']['physical_api_calls'],2)
        self.assertEqual(summary['advisor']['unknown_cost_calls'],1)
        self.assertEqual(summary['advisor']['observed_cost_usd'],'0.02')
        self.assertIsNone(summary['total_cost_usd'])

    def test_exhausted_routes_fail_before_next_actor_call_and_retain_patch(self):
        agent=self.agent([response(1,args={'command':"printf 'candidate\\n' > tracked.txt",'state':STATE}),response(2)],k=2,
                         advisor=self.routed_advisor(fail_both=True))
        with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['stop_reason'],'AdvisorUnavailable')
        self.assertEqual(summary['actor']['api_calls'],2)
        self.assertEqual(summary['actor']['cost_usd'],'0.002')
        self.assertEqual(summary['advisor']['physical_api_calls'],2)
        self.assertEqual(summary['advisor']['observed_cost_usd'],'0.003')
        self.assertEqual(summary['advisor']['unknown_cost_calls'],1)
        self.assertIn('+candidate',agent.patch_text)
        self.assertEqual(agent.n_calls,2)

    def test_production_actor_factory_preserves_routes_and_tool_payload(self):
        actor=StructuredActorModel(model_name='openrouter/deepseek/mock',model_kwargs={
            'drop_params':True,'max_tokens':512,'timeout':30,'reasoning_effort':'medium',
            'extra_body':{'provider':{'only':['DeepInfra'],'allow_fallbacks':False}}})
        from autoresearch_budget.store import RunStore
        actor.bind(RunStore(self.root/'factory',run_id='factory'))
        with patch.dict('os.environ',{'OPENROUTER_API_KEY':'offline-key'}):
            client=actor._client()
        captured=[]
        def transport(request,timeout):
            captured.append(json.loads(request.data)); self.assertEqual(timeout,30)
            return 200,json.dumps(response(1)).encode()
        client.transport=transport; actor.actor_call=1
        message=actor.query([{'role':'user','content':'issue'}])
        body=captured[0]
        self.assertEqual(body['model'],'deepseek/mock')
        self.assertEqual(body['provider']['only'],['DeepInfra'])
        self.assertEqual(body['reasoning'],{'effort':'medium'})
        self.assertEqual(body['max_tokens'],512)
        self.assertEqual(body['tool_choice'],'required')
        self.assertEqual({tool['function']['name'] for tool in body['tools']},{'execute','submit'})
        self.assertNotIn('drop_params',body)
        self.assertNotIn('parallel_tool_calls',body)
        self.assertEqual(message['extra']['actions'][0]['name'],'execute')

    def test_full_output_is_retained_and_shell_sentinel_does_not_submit(self):
        command="printf 'COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\\n'; printf '%0500d' 0"
        agent=self.agent([response(1,args={'command':command,'state':STATE}),response(2,name='submit',args={'summary':'done','validation_refs':['observation-1'],'state':STATE})])
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.n_calls,2)
        events=self.events(agent)
        raw=next(e for e in events if e['kind']=='raw_observation')['payload']['result']['output']
        visible=next(e for e in events if e['kind']=='visible_observation')['payload']['messages'][0]
        self.assertGreater(len(raw),500)
        self.assertLess(len(visible['content']),300)
        self.assertTrue(visible['extra']['output_truncated'])
        self.assertIn('full output retained',visible['content'])

    def test_builtin_config_merge_loads_custom_classes_and_consistent_prompts(self):
        config=recursive_merge(get_config_from_spec('mini'),{
            'agent':{'agent_class':'autoresearch_budget.advisory_agent.AdvisoryAgent','cost_limit':0,
                     'advice':{'policy':'fixed_interval'},'output_path':self.root/'trajectory.json'},
            'model':{'model_class':'autoresearch_budget.actor_model.StructuredActorModel','model_name':'openrouter/mock'},
            'environment':{'environment_class':'autoresearch_budget.poc_environment.PocEnvironment','cwd':str(self.repo)}})
        agent=get_agent(get_model(config=config['model']),get_environment(config['environment']),config['agent'])
        self.assertIsInstance(agent,AdvisoryAgent)
        self.assertNotIn('bash tool',agent.config.instance_template)
        self.assertNotIn('COMPLETE_TASK_AND_SUBMIT',agent.config.instance_template)
        self.assertIn('execute or submit',agent.config.system_template)


@unittest.skipUnless(importlib.util.find_spec('harbor') is not None, 'Harbor runtime is optional')
class HarborAdapterTest(unittest.TestCase):
    def test_installer_body_and_extension_uploads(self):
        import asyncio
        from autoresearch_budget.harbor_mini_agent import BudgetMiniSweAgent, AdvisoryMiniSweAgent
        with TemporaryDirectory() as temp:
            adapter=object.__new__(AdvisoryMiniSweAgent)
            adapter._version='2.4.6'; adapter.logs_dir=Path(temp)
            adapter.ensure_system_dependencies=AsyncMock()
            adapter.exec_as_agent=AsyncMock(); adapter.exec_as_root=AsyncMock()
            environment=Mock(); environment.upload_file=AsyncMock()
            asyncio.run(adapter.install(environment))
            command=adapter.exec_as_agent.call_args.kwargs['command']
            self.assertIn('mini-swe-agent==2.4.6',command)
            self.assertIn('uv tool dir --bin',command)
            names={call.args[0].name for call in environment.upload_file.call_args_list}
            self.assertIn('actor_model.py',names); self.assertIn('contracts.py',names)
            self.assertEqual(set(json.loads((Path(temp)/'poc-source.json').read_text())),names)


if __name__=='__main__':
    unittest.main()
