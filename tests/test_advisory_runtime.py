"""Offline integration tests against mini-SWE 2.4.6; no provider calls."""
import importlib.util
import json
import subprocess
import time
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

    def agent(self,responses,*,limit=80,k=10,advisor_content=None,advisor=None,
              policy='fixed_interval',max_calls=100,max_context_chars=60000,wall_time_limit_seconds=0,
              production_template=False, **policy_options):
        self.requests=[]
        queue=list(responses)
        def transport(request, timeout):
            body=json.loads(request.data); self.requests.append(body)
            if not queue:
                raise AssertionError('Unexpected model call')
            return 200,json.dumps(queue.pop(0)).encode()
        formatting={'observation_template':get_config_from_spec('mini')['model']['observation_template']}
        if not production_template:formatting={'observation_template':'{{ output.output }}'}
        actor=StructuredActorModel(model_name='openrouter/mock/actor',backend=OpenRouterClient(
            api_key='offline-test-key',model='mock/actor',allow_paid=True,transport=transport),
            **formatting)
        env=PocEnvironment(cwd=str(self.repo),env={'OPENROUTER_API_KEY':'','MSWEA_API_KEY':''})
        if advisor is None:
            def advice_transport(request,timeout):
                return 200,json.dumps({'id':'advice-model','model':'mock/advisor','choices':[{'message':{
                    'content':advisor_content if advisor_content is not None else json.dumps(GUIDANCE)},'finish_reason':'stop'}],
                    'usage':{'cost':0.02,'prompt_tokens':20,'completion_tokens':10}}).encode()
            advisor=Advisor(OpenRouterClient(api_key='offline-test-key',model='mock/advisor',allow_paid=True,
                                           transport=advice_transport),128)
        advice={'policy':policy,'max_calls':max_calls,'max_context_chars':max_context_chars,'run_id':'offline-task',**policy_options}
        if policy == 'fixed_interval':advice['every_steps']=k
        agent=AdvisoryAgent(actor,env,advice=advice,advisor=advisor,
            cost_limit=0,step_limit=limit,system_template='ignored',instance_template='ignored',
            wall_time_limit_seconds=wall_time_limit_seconds,
            output_path=self.root/'agent'/'mini-swe-agent.trajectory.json',max_observation_chars=200)
        actor.backend.tools=__import__('autoresearch_budget.contracts',fromlist=['actor_tools']).actor_tools()
        actor.backend.recorder=actor._record
        if hasattr(advisor,'backend'):
            advisor.backend.recorder=agent._advisor_record
            if hasattr(advisor.backend,'primary'):
                advisor.backend.primary.recorder=advisor.backend.fallback.recorder=agent._advisor_record
        return agent

    def request(self, index, *, ids=None, **extra):
        return response(index, name='request_advice', args={
            'question':'Which hypothesis fits the observed failure?',
            'evidence_ids':ids or ['observation-1'], 'state':STATE, **extra})

    def finish(self, index, **extra):
        return response(index, name='submit', args={'summary':'done', 'validation_refs':['observation-1'],
                                                  'state':STATE, **extra})

    def plan(self, index, *, hypothesis="A consumer retains stale state", **extra):
        return response(index, name="propose_plan", args={
            "hypothesis": hypothesis, "approach": "Invalidate the consumer cache",
            "falsifying_check": "Disable cache and reproduce", "validation_plan": "Test both entry points",
            "previous_plan_id": "", "change_reason": "The observed consumer behavior supports this initial plan",
            "evidence_ids": ["observation-1"], "state": STATE, **extra})

    def checkpoint(self, index, *, question="Why does the same check still fail?", **extra):
        return response(index, name="report_checkpoint", args={
            "question": question, "next_check": "Run a discriminating regression",
            "evidence_ids": ["observation-1"], "state": STATE, **extra})

    def failed_check(self, index, *, kind="semantic", expected="success", **extra):
        return response(index, args={"command": "printf 'assertion failed'; exit 1",
            "check": {"id": "regression", "kind": kind, "expected": expected},
            "state": {**STATE, "phase": "diagnose"}, **extra})

    def test_phase3_plan_review_is_before_execution_and_uses_public_evidence(self):
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(4, advice_progress=[progress("advice-1", "observation-3")])],
            policy="plan_review", max_calls=7, production_template=True)
        self.assertEqual(agent.run("Public issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        tool = next(m for m in self.requests[2]["messages"] if m.get("tool_call_id") == "tool-2")
        self.assertIn("Advisor guidance", json.loads(tool["content"])["output"])
        packet = next(e["payload"]["packet"] for e in self.events(agent) if e["kind"] == "advice_requested")
        self.assertEqual(packet["reason"], "plan_review")
        self.assertIn("Invalidate the consumer cache", packet["question"])
        self.assertEqual(set(packet["cited_evidence"]), {"observation-1"})
        self.assertEqual(agent.state["hypothesis"], "A consumer retains stale state")
        self.assertEqual(json.loads((agent.advice_store.run_dir/"metadata.json").read_text())["contract_version"], "phase3-v3")

    def test_phase3_followup_keeps_prior_advice_open_and_requires_all_resolutions(self):
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.request(4, ids=["observation-3"], advice_progress=[progress("advice-1", "", status="in_progress")]),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-2"),
                              "advice_progress": [progress("advice-1", "", status="in_progress")]}),
            self.finish(6, advice_progress=[progress("advice-2", "observation-5")]),
            self.finish(7, advice_progress=[progress("advice-1", "", status="in_progress"), progress("advice-2", "observation-5")]),
            self.finish(8, advice_progress=[progress("advice-1", "observation-5"), progress("advice-2", "observation-5")])],
            policy="plan_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 2)
        packet = [e["payload"]["packet"] for e in self.events(agent) if e["kind"] == "advice_requested"][1]
        self.assertEqual(packet["related_advice_ids"], ["advice-1"])
        self.assertEqual(packet["new_observation_ids"], ["observation-3"])
        history = json.loads(packet["advice_followthrough"])
        self.assertEqual(history[0]["status"], "in_progress")
        self.assertEqual(history[0]["guidance"], GUIDANCE)
        self.assertEqual(agent.advice_records["advice-2"]["related_advice_ids"], ["advice-1"])
        self.assertTrue(all(r["status"] == "validated" for r in agent.advice_records.values()))
        errors = [e["payload"]["error"] for e in self.events(agent) if e["kind"] == "actor_schema_error"]
        self.assertTrue(any("exactly the open advice" in error for error in errors))
        self.assertTrue(any("before submission" in error for error in errors))

    def test_phase3_plan_revision_can_review_while_prior_advice_is_in_progress(self):
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.plan(4, hypothesis="The producer fails to update", previous_plan_id="plan-1",
                      change_reason="The fresh check shifts the causal explanation", evidence_ids=["observation-3"],
                      advice_progress=[progress("advice-1", "", status="in_progress")]),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-2"),
                              "advice_progress": [progress("advice-1", "", status="in_progress")]}),
            self.finish(6, advice_progress=[progress("advice-1", "observation-5"), progress("advice-2", "observation-5")])],
            policy="plan_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 2)
        self.assertEqual(agent.interventions.plan_number, 2)
        packet = [e["payload"]["packet"] for e in self.events(agent) if e["kind"] == "advice_requested"][1]
        self.assertEqual(packet["related_advice_ids"], ["advice-1"])
        self.assertEqual(json.loads(packet["advice_followthrough"])[0]["status"], "in_progress")

    def test_phase3_pending_guidance_still_blocks_followup_and_automatic_dispatch(self):
        agent = self.agent([response(1), self.plan(2), self.request(3), self.plan(4),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(6, advice_progress=[progress("advice-1", "observation-5")])], policy="plan_review")
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        errors = [e["payload"]["error"] for e in self.events(agent) if e["kind"] == "actor_schema_error"]
        self.assertEqual(sum("pending guidance" in e for e in errors), 2)
        agent.pending_advice = "advice-1"
        self.assertIn("pending_advice", agent._phase3_consult("Clarify", ["plan_review"], ["observation-1"]))
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(agent.pending_advice, "advice-1")

    def test_phase3_followup_limit_does_not_close_prior_advice(self):
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.request(4, advice_progress=[progress("advice-1", "", status="in_progress")]),
            self.finish(5, advice_progress=[progress("advice-1", "observation-3")])],
            policy="plan_review", max_calls=1)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(agent.advice_denials, {"call_limit": 1})
        result = [e["payload"] for e in self.events(agent) if e["kind"] == "intervention_result"][-1]
        self.assertFalse(result["delivered"])
        packet_history = [e["payload"] for e in self.events(agent) if e["kind"] == "advice_status"]
        self.assertTrue(any(e.get("status") == "in_progress" for e in packet_history))

    def test_phase3_failed_followup_preserves_earlier_open_status(self):
        advisor = Mock()
        advisor.consult.side_effect = [Completion(content=json.dumps(GUIDANCE), response_id="offline", requested_model="mock/advisor", served_model="mock/advisor", cost_usd=Decimal("0.02")),
                                      ModelError("Follow-up unavailable")]
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.request(4, advice_progress=[progress("advice-1", "", status="in_progress")]),
            self.finish(5, advice_progress=[progress("advice-1", "observation-3")])],
            policy="plan_review", advisor=advisor, max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 2)
        self.assertEqual(agent.advisor_failures, 1)
        self.assertEqual(set(agent.advice_records), {"advice-1"})
        self.assertIsNone(agent.pending_advice)
        packet = advisor.consult.call_args_list[1].args[0]
        self.assertEqual(json.loads(packet["advice_followthrough"])[0]["status"], "in_progress")
        statuses = [e["payload"]["status"] for e in self.events(agent) if e["kind"] == "advice_status"]
        self.assertEqual(statuses[-2:], ["in_progress", "validated"])

    def test_phase3_hybrid_recovery_can_review_while_plan_advice_is_open(self):
        ongoing = {"advice_progress": [progress("advice-1", "", status="in_progress")]}
        agent = self.agent([response(1), self.plan(2),
            self.failed_check(3, advice_decision=adopt("advice-1")),
            self.failed_check(4, **ongoing), self.failed_check(5, **ongoing),
            self.checkpoint(6, evidence_ids=["observation-5"], **ongoing),
            response(7, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-2"), **ongoing}),
            self.finish(8, advice_progress=[progress("advice-1", "observation-7"), progress("advice-2", "observation-7")])],
            policy="hybrid", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        requests = [e["payload"] for e in self.events(agent) if e["kind"] == "advice_requested"]
        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[1]["reason"], "repeated_failure")
        self.assertEqual(requests[1]["related_advice_ids"], ["advice-1"])
        self.assertEqual(requests[1]["packet"]["new_observation_ids"], ["observation-5"])

    def test_phase3_control_has_common_tools_without_automatic_reviews(self):
        agent = self.agent([response(1), self.plan(2), self.finish(3)], policy="phase3_control", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 0)
        self.assertEqual(agent.interventions.plan_number, 1)
        self.assertEqual({t["function"]["name"] for t in self.requests[0]["tools"]},
                         {"execute", "submit", "request_advice", "propose_plan", "report_checkpoint"})

    def test_phase3_plan_deduplication_and_material_revision(self):
        reject = lambda id: {"advice_id": id, "decision": "reject", "reason": "Public evidence contradicts it",
                             "evidence_refs": ["observation-1"]}
        agent = self.agent([response(1), self.plan(2),
            response(3, args={"command": "true", "state": STATE, "advice_decision": reject("advice-1")}),
            self.plan(4), self.plan(5, hypothesis="The producer fails to update", previous_plan_id="plan-1",
                                    change_reason="The fresh observation contradicts the consumer explanation"),
            response(6, args={"command": "true", "state": STATE, "advice_decision": reject("advice-2")}),
            self.finish(7)], policy="plan_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 2)
        self.assertEqual(agent.interventions.plan_number, 2)
        packet = [e["payload"]["packet"] for e in self.events(agent) if e["kind"] == "advice_requested"][1]
        self.assertIn('"previous_plan": {"plan_id": "plan-1"', packet['question'])
        self.assertIn('contradicts the consumer explanation', packet['question'])
        self.assertIn('advice_followthrough', packet)

    def test_phase3_inspection_accepts_insufficient_evidence_and_preserves_isolation(self):
        inspection = {"explanation": "Insufficient evidence to distinguish producer and consumer",
                      "approach": "Inspect the cache boundary", "alternatives": "Producer or consumer",
                      "missing_evidence": "A cache bypass result"}
        test_root = self.root
        for mode in ("plan_review", "hybrid", "recovery_review", "phase3_control"):
            with self.subTest(mode=mode):
                self.root = test_root/mode
                actions = [response(1), response(2), self.checkpoint(3, inspection=inspection)]
                if mode in {"plan_review", "hybrid"}:
                    actions += [response(4, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
                                self.finish(5, advice_progress=[progress("advice-1", "observation-4")])]
                else:
                    actions += [self.finish(4)]
                agent = self.agent(actions, policy=mode, inspection_steps=2, max_calls=7)
                self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
                self.assertEqual(agent.advisor_calls, int(mode in {"plan_review", "hybrid"}))
                events = self.events(agent)
                self.assertEqual(next(e['payload']['actor_call'] for e in events if e['kind'] == 'inspection_checkpoint'), 2)
                self.assertEqual(agent.interventions.inspection_responses[0]['inspection'], inspection)
                self.assertIsNone(agent.interventions.plan)
                if agent.advisor_calls:
                    packet = next(e['payload']['packet'] for e in events if e['kind'] == 'advice_requested')
                    self.assertEqual(packet['reason'], 'inspection_checkpoint')
                    self.assertIn('Insufficient evidence', packet['question'])

    def test_phase3_inspection_cannot_be_skipped_or_answered_without_reasoning(self):
        inspection = {key: 'Unknown; inspect boundary next' for key in ('explanation', 'approach', 'alternatives', 'missing_evidence')}
        agent = self.agent([response(1), response(2),
            response(3, args={'command': 'touch checkpoint-bypassed', 'state': STATE}),
            self.finish(4), self.checkpoint(5, inspection=inspection), self.finish(6)],
            policy='phase3_control', inspection_steps=2)
        self.assertEqual(agent.run('Issue')['exit_status'], 'Submitted')
        self.assertFalse((self.repo/'checkpoint-bypassed').exists())
        self.assertEqual(len([e for e in self.events(agent) if e['kind'] == 'actor_schema_error']), 2)

    def test_phase3_inspection_requires_reasoning_fields(self):
        inspection = {key: 'Unknown; inspect boundary next' for key in ('explanation', 'approach', 'alternatives', 'missing_evidence')}
        agent = self.agent([response(1), response(2), self.checkpoint(3),
                            self.checkpoint(4, inspection=inspection), self.finish(5)],
                           policy='phase3_control', inspection_steps=2)
        self.assertEqual(agent.run('Issue')['exit_status'], 'Submitted')
        self.assertEqual(len([e for e in self.events(agent) if e['kind'] == 'actor_schema_error']), 1)

    def test_phase3_plan_satisfies_inspection_with_one_combined_consultation(self):
        agent = self.agent([response(1), response(2), self.plan(3),
            response(4, args={'command': 'true', 'state': STATE, 'advice_decision': adopt('advice-1')}),
            self.finish(5, advice_progress=[progress('advice-1', 'observation-4')])],
            policy='plan_review', inspection_steps=2)
        self.assertEqual(agent.run('Issue')['exit_status'], 'Submitted')
        self.assertEqual(agent.advisor_calls, 1)
        reasons = next(e['payload']['reasons'] for e in self.events(agent) if e['kind'] == 'intervention')
        self.assertEqual(reasons, ['plan_review', 'inspection_checkpoint'])
        self.assertIsNone(agent.interventions.inspection_checkpoint)

    def test_phase3_revision_rejects_stale_predecessor_before_review(self):
        agent = self.agent([response(1), self.plan(2),
            self.plan(3, hypothesis='Producer is responsible', previous_plan_id='plan-0'),
            self.plan(4, hypothesis='Producer is responsible', previous_plan_id='plan-1'), self.finish(5)],
            policy='phase3_control')
        self.assertEqual(agent.run('Issue')['exit_status'], 'Submitted')
        self.assertEqual(agent.interventions.plan_number, 2)
        self.assertEqual(agent.interventions.plans[1]['previous_plan_id'], 'plan-1')
        self.assertEqual(len([e for e in self.events(agent) if e['kind'] == 'actor_schema_error']), 1)

    def test_phase3_repeated_failure_review_and_isolation(self):
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3), self.checkpoint(4),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(6, advice_progress=[progress("advice-1", "observation-5")])],
            policy="recovery_review", max_calls=7, inspection_steps=100)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        events = self.events(agent)
        self.assertEqual([e["payload"]["failure_streak"] for e in events if e["kind"] == "check_result"], [1, 2, 3])
        self.assertEqual(next(e["payload"]["reasons"] for e in events if e["kind"] == "recovery_checkpoint"), ["repeated_failure"])
        self.assertEqual(next(e["payload"]["reason"] for e in events if e["kind"] == "advice_requested"), "repeated_failure")

    def test_phase3_plan_only_does_not_dispatch_recovery_advice(self):
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3),
                            self.checkpoint(4), self.finish(5)], policy="plan_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 0)
        self.assertTrue(any(e["kind"] == "recovery_checkpoint" for e in self.events(agent)))

    def test_phase3_recovery_only_records_plan_and_retains_actor_requests(self):
        agent = self.agent([response(1), self.plan(2), self.request(3),
            response(4, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(5, advice_progress=[progress("advice-1", "observation-4")])],
            policy="recovery_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(next(e["payload"]["reason"] for e in self.events(agent) if e["kind"] == "advice_requested"), "advice_requested")

    def test_phase3_recovery_does_not_repeat_on_unchanged_failure_evidence(self):
        reject = {"advice_id": "advice-1", "decision": "reject", "reason": "The check contradicts the suggestion",
                  "evidence_refs": ["observation-1"]}
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3), self.checkpoint(4),
            self.failed_check(5, advice_decision=reject), self.failed_check(6), self.failed_check(7),
            self.finish(8)], policy="recovery_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(len([e for e in self.events(agent) if e["kind"] == "recovery_checkpoint"]), 1)

    def test_phase3_skipped_checkpoint_still_reviews_after_one_command(self):
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3), self.failed_check(4),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(6, advice_progress=[progress("advice-1", "observation-5")])], policy="recovery_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(next(e["payload"]["actor_call"] for e in self.events(agent) if e["kind"] == "advice_requested"), 4)

    def test_phase3_stall_checkpoint_allows_one_experiment_before_review(self):
        diagnose = {**STATE, "phase": "diagnose"}
        agent = self.agent([*[response(i, args={"command": "true", "state": diagnose}) for i in range(1, 10)],
            self.checkpoint(10, question=""), response(11, args={"command": "true", "state": diagnose}),
            response(12, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(13, advice_progress=[progress("advice-1", "observation-12")])],
            policy="recovery_review", max_calls=7, inspection_steps=100)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        events = self.events(agent)
        self.assertEqual(next(e["payload"]["actor_call"] for e in events if e["kind"] == "recovery_checkpoint"), 9)
        self.assertEqual(next(e["payload"]["actor_call"] for e in events if e["kind"] == "advice_requested"), 11)
        self.assertEqual(agent.advisor_calls, 1)

    def test_phase3_combines_plan_and_recovery_reasons(self):
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3), self.plan(4),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(6, advice_progress=[progress("advice-1", "observation-5")])], policy="hybrid", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        reasons = next(e["payload"]["reasons"] for e in self.events(agent) if e["kind"] == "intervention")
        self.assertEqual(reasons, ["plan_review", "repeated_failure"])

    def test_phase3_actor_request_merges_recovery_and_shares_ceiling(self):
        agent = self.agent([self.failed_check(1), self.failed_check(2), self.failed_check(3), self.request(4),
            response(5, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.plan(6, advice_progress=[progress("advice-1", "observation-5")]), self.finish(7)],
            policy="hybrid", max_calls=1)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(agent.advice_denials, {"call_limit": 1})
        first = next(e["payload"] for e in self.events(agent) if e["kind"] == "intervention")
        self.assertEqual(first["reasons"], ["advice_requested", "repeated_failure"])

    def test_phase3_uncertainty_requires_question_and_visible_evidence(self):
        state = {**STATE, "phase": "diagnose", "uncertainty": "high", "blocker": "missing_information",
                 "uncertainty_question": "Does the cache explain the mismatch?", "evidence_refs": ["observation-1"]}
        agent = self.agent([response(1), *[response(i, args={"command": "true", "state": state}) for i in range(2, 5)],
            self.checkpoint(5), response(6, args={"command": "true", "state": STATE, "advice_decision": adopt("advice-1")}),
            self.finish(7, advice_progress=[progress("advice-1", "observation-6")])], policy="recovery_review", max_calls=7)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(next(e["payload"]["reasons"] for e in self.events(agent) if e["kind"] == "recovery_checkpoint"), ["persistent_uncertainty"])

    def test_recovery_v1_policy_selection_and_contract_isolation(self):
        from autoresearch_budget.interventions import RecoveryCollector, InterventionCollector
        base = self.root
        for mode in ('recovery_review','hybrid','plan_review','phase3_control'):
            with self.subTest(mode=mode):
                self.root=base/mode
                agent=self.agent([response(1),self.finish(2)],policy=mode)
                self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
                recovery=mode in ('recovery_review','hybrid')
                self.assertEqual(type(agent.interventions),RecoveryCollector if recovery else InterventionCollector)
                metadata=json.loads((agent.advice_store.run_dir/'metadata.json').read_text())
                self.assertEqual(metadata['contract_version'],'phase3-recovery-v1' if recovery else 'phase3-v3')
                fields=self.requests[0]['tools'][0]['function']['parameters']['properties']['state']['properties']
                self.assertEqual('concern_id' in fields,recovery)
        self.root=base

    def test_recovery_v1_noise_and_interleaved_check_preserve_failure(self):
        command="printf 'AssertionError: expected 2 got 1\\n1 failed in '; date +'%S seconds'; exit 1"
        actions=[self.failed_check(1,command=command),self.failed_check(2,command=command),
            response(3,args={'command':'true','state':STATE,'check':{'id':'other','kind':'semantic','expected':'success'}}),
            self.failed_check(4,command=command),self.checkpoint(5),
            response(6,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(7,advice_progress=[progress('advice-1','observation-6')])]
        agent=self.agent(actions,policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        records=[e['payload'] for e in self.events(agent) if e['kind']=='check_result']
        self.assertEqual([r['failure_streak'] for r in records],[1,2,0,3])
        self.assertTrue(all(r['detector_version']=='phase3-recovery-v1' for r in records))

    def test_recovery_v1_deferred_target_pass_resolves_without_advice(self):
        agent=self.agent([self.failed_check(1),self.failed_check(2),self.failed_check(3),
            self.checkpoint(4,question='None (run the regression first)'),
            self.failed_check(5),self.finish(6)],policy='recovery_review',inspection_steps=100)
        # Simulate the same declared check returning a successful result.
        original=agent.env.execute
        def execute(action):
            if action.get('observation_id')=='observation-5':
                return {'returncode':0,'output':'Passed','exception_info':''}
            return original(action)
        agent.env.execute=execute
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,0)
        self.assertEqual(agent.interventions.decisions[-1]['disposition'],'resolved_by_experiment')

    def test_recovery_v1_unrelated_pass_still_dispatches_deferred_review(self):
        agent=self.agent([self.failed_check(1),self.failed_check(2),self.failed_check(3),
            self.checkpoint(4,question='None (next test can discriminate)'),
            response(5,args={'command':'true','state':STATE,'check':{'id':'other','kind':'semantic','expected':'success'}}),
            response(6,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(7,advice_progress=[progress('advice-1','observation-6')])],
            policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        requested=next(e['payload'] for e in self.events(agent) if e['kind']=='advice_requested')
        self.assertEqual(requested['actor_call'],5)

    def test_recovery_v1_missing_support_is_local_then_counts_paraphrases(self):
        high={**STATE,'phase':'diagnose','uncertainty':'high'}
        actions=[response(1,args={'command':'true','state':high}),
            self.checkpoint(2,question='Which cause fits?',state=high),
            response(3,args={'command':'true','state':{**high,'uncertainty_question':'Why does the cache retain state?',
                                                     'evidence_refs':['observation-1']}}),
            response(4,args={'command':'true','state':{**high,'uncertainty_question':'Why is invalidation ineffective?',
                                                     'evidence_refs':['observation-1']}}),
            self.checkpoint(5),
            response(6,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(7,advice_progress=[progress('advice-1','observation-6')])]
        agent=self.agent(actions,policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        checkpoints=[e['payload'] for e in self.events(agent) if e['kind']=='recovery_checkpoint']
        self.assertEqual([c['reasons'] for c in checkpoints],[['concern_evidence_required'],['persistent_uncertainty']])
        self.assertEqual([c['concern_id'] for c in checkpoints],['concern-1','concern-1'])
        self.assertEqual(agent.advisor_calls,1)
        self.assertEqual(next(e['payload']['actor_call'] for e in self.events(agent) if e['kind']=='advice_requested'),5)

    def test_recovery_v1_mixed_stall_nomination_accepts_support(self):
        diagnose={**STATE,'phase':'diagnose'}
        high={**diagnose,'uncertainty':'high'}
        actions=[response(i,args={'command':'true','state':diagnose}) for i in range(1,9)]
        actions += [response(9,args={'command':'true','state':high}),self.checkpoint(10,state=high),
            response(11,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(12,advice_progress=[progress('advice-1','observation-11')])]
        agent=self.agent(actions,policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        self.assertEqual(next(e['payload']['reason'] for e in self.events(agent) if e['kind']=='advice_requested'),'stall')

    def test_recovery_v1_unavailable_review_does_not_retry_same_failure(self):
        agent=self.agent([self.failed_check(1),self.failed_check(2),self.failed_check(3),self.checkpoint(4),
            self.failed_check(5),self.failed_check(6),self.finish(7)],policy='recovery_review',inspection_steps=100,
            advisor=self.routed_advisor(fail_both=True))
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        self.assertEqual(agent.advisor_failures,1)
        self.assertEqual(agent.interventions.failure_count,5)

    def test_recovery_v1_explicit_evidence_resolves_contradiction_without_advice(self):
        conflict={**STATE,'phase':'diagnose','blocker':'conflicting_evidence',
                  'uncertainty_question':'Why does the cache retain old state?', 'evidence_refs':['observation-1']}
        agent=self.agent([response(1),response(2,args={'command':'true','state':conflict}),
            response(3,args={'command':'true','state':{**conflict,'uncertainty_question':'Why is invalidation ineffective?'}}),
            self.checkpoint(4,question=''),
            response(5,args={'command':'true','state':{**STATE,'concern_id':'concern-1',
                        'concern_status':'resolved','evidence_refs':['observation-3']}}),self.finish(6)],
            policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,0)
        self.assertEqual(next(e['payload']['reasons'] for e in self.events(agent) if e['kind']=='recovery_checkpoint'),
                         ['conflicting_evidence'])
        self.assertIsNone(agent.interventions.concern)
        self.assertEqual(agent.interventions.concern_history[-1]['resolution_evidence'],['observation-3'])

    def test_recovery_v1_environment_blocker_defers_pending_review(self):
        environment={**STATE,'phase':'diagnose','blocker':'environment'}
        agent=self.agent([self.failed_check(1),self.failed_check(2),self.failed_check(3),
            self.checkpoint(4,state=environment),response(5,args={'command':'true','state':environment}),
            self.finish(6,state=environment)],policy='recovery_review',inspection_steps=100)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,0)
        self.assertTrue(any(e['kind']=='recovery_deferred' and e['payload']['reason']=='environment_blocker'
                            for e in self.events(agent)))

    def test_phase3_optional_advisor_failure_continues_and_keeps_usage(self):
        agent = self.agent([response(1), self.plan(2), self.finish(3)], policy="hybrid", max_calls=7,
                           advisor=self.routed_advisor(fail_both=True))
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 1)
        self.assertEqual(agent.advisor_failures, 1)
        self.assertEqual(agent.advisor_api_calls, 2)
        self.assertIsNone(agent.pending_advice)
        self.assertEqual(agent.advisor_cost, Decimal("0.003"))

    def test_phase3_late_plan_denied_without_dispatch(self):
        agent = self.agent([response(1), self.plan(2), self.finish(3)], policy="hybrid", max_calls=7, limit=5)
        self.assertEqual(agent.run("Issue")["exit_status"], "Submitted")
        self.assertEqual(agent.advisor_calls, 0)
        self.assertEqual(agent.advice_denials, {"no_actor_capacity": 1})

    def test_command_deadline_is_bound_to_agent_wall_time(self):
        agent=self.agent([response(1)],wall_time_limit_seconds=5)
        agent._start_time=time.time()-6
        output=agent.env.execute({'command':'touch deadline-overrun'})
        self.assertTrue(output['timed_out'])
        self.assertFalse((self.repo/'deadline-overrun').exists())

    def test_command_record_identifies_patch_at_launch(self):
        agent=self.agent([response(1),self.finish(2)],policy='actor_only')
        agent.run('Offline task')
        commands=[json.loads(line)['payload'] for line in agent.advice_store.events_path.read_text().splitlines()
                  if json.loads(line)['kind']=='command']
        command=next(record for record in commands if record['observation_id']=='observation-1')
        self.assertEqual(command['patch_id'],'patch-1')
        self.assertFalse(command['timed_out'])

    def test_requested_policy_never_consults_automatically(self):
        agent=self.agent([*[response(i) for i in range(1,26)],self.finish(26)],
                         policy='advice_requested',max_calls=7)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,0)
        self.assertTrue(all('request_advice' in {t['function']['name'] for t in r['tools']} for r in self.requests))
        self.assertIn('7 maximum; 7 remaining', self.requests[0]['messages'][-1]['content'])
        metadata=json.loads((agent.advice_store.run_dir/'metadata.json').read_text())
        self.assertEqual(metadata['contract_version'],'phase2-v1')
        self.assertNotIn('arrives every',agent.config.system_template)
        self.assertIn('after initial inspection',self.requests[0]['messages'][0]['content'])
        self.assertIn('If two checks fail',self.requests[0]['messages'][0]['content'])
        self.assertIn('there is no minimum number of calls',self.requests[0]['messages'][0]['content'])

    def test_requested_advice_matches_tool_response_and_tracks_followthrough(self):
        agent=self.agent([response(1),self.request(2,ids=['observation-1','patch-1']),
            response(3,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(4,advice_progress=[progress('advice-1','observation-3')])],
            policy='advice_requested',max_calls=7)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.n_calls,4)
        self.assertEqual(agent.advice_records['advice-1']['status'],'validated')
        tools=[m for m in self.requests[2]['messages'] if m.get('tool_call_id')=='tool-2']
        self.assertEqual(len(tools),1); self.assertEqual(tools[0]['role'],'tool')
        self.assertIn('Advisor guidance (advice-1)',tools[0]['content'])
        packet=next(e['payload']['packet'] for e in self.events(agent) if e['kind']=='advice_requested')
        self.assertEqual(packet['question'],'Which hypothesis fits the observed failure?')
        self.assertEqual(packet['actor_calls_remaining'],78)
        self.assertEqual(set(packet['cited_evidence']),{'observation-1','patch-1'})
        self.assertEqual([e['payload']['actor_call'] for e in self.events(agent) if e['kind']=='raw_observation'],[1,3])
        self.assertIn('1 used / 7 maximum; 6 remaining', self.requests[2]['messages'][-1]['content'])

    def test_requested_ceiling_denies_without_removing_tool(self):
        reject={'advice_id':'advice-1','decision':'reject','reason':'Existing evidence rules it out',
                'evidence_refs':['observation-1']}
        agent=self.agent([response(1),self.request(2),response(3,args={'command':'true','state':STATE,
            'advice_decision':reject}),self.request(4),self.finish(5)],policy='advice_requested',max_calls=1)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        self.assertEqual(agent.advice_requests,2)
        self.assertEqual(agent.advice_denials,{'call_limit':1})
        self.assertIn('call_limit',next(m['content'] for m in self.requests[4]['messages'] if m.get('tool_call_id')=='tool-4'))
        self.assertIn('request_advice',{t['function']['name'] for t in self.requests[4]['tools']})

    def test_requested_production_template_formats_guidance_and_denial(self):
        agent=self.agent([response(1),self.request(2),
            response(3,args={'command':'touch unacknowledged','state':STATE}),
            response(4,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.request(5,advice_progress=[progress('advice-1',None,status='in_progress')]),
            self.finish(6,advice_progress=[progress('advice-1','observation-4')])],
            policy='advice_requested',max_calls=1,production_template=True)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advice_denials,{'call_limit':1})
        for index,call_id,expected in ((2,'tool-2','Advisor guidance (advice-1)'),(5,'tool-5','call_limit')):
            tool=next(m for m in self.requests[index]['messages'] if m.get('tool_call_id')==call_id)
            observation=json.loads(tool['content'])
            self.assertEqual(observation['returncode'],0)
            self.assertIn(expected,observation['output'])
        self.assertIn('advice_decision for advice-1 is required', self.requests[3]['messages'][-2]['content'])
        self.assertFalse((self.repo/'unacknowledged').exists())
        self.assertEqual(agent.n_calls,6)
        self.assertEqual(agent.advice_records['advice-1']['status'],'validated')

    def test_requested_zero_limit_and_last_turn_do_not_dispatch(self):
        for ceiling,limit,reason in ((0,3,'call_limit'),(7,2,'no_actor_capacity')):
            with self.subTest(ceiling=ceiling,limit=limit):
                actions=[response(1),self.request(2)]
                if limit==3:actions.append(self.finish(3))
                agent=self.agent(actions,policy='advice_requested',max_calls=ceiling,limit=limit)
                self.assertEqual(agent.run('Issue')['exit_status'],'Submitted' if limit==3 else 'LimitsExceeded')
                self.assertEqual(agent.advisor_calls,0)
                self.assertEqual(agent.advice_denials,{reason:1})

    def test_requested_unknown_evidence_and_pending_guidance_are_rejected(self):
        agent=self.agent([response(1),self.request(2,ids=['observation-999']),
            self.request(3,ids=['patch-999']),self.request(4),self.request(5),
            response(6,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(7,advice_progress=[progress('advice-1','observation-6')])],policy='advice_requested')
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,1)
        errors=[e['payload']['error'] for e in self.events(agent) if e['kind']=='actor_schema_error']
        self.assertEqual(len(errors),3)
        self.assertIn('unseen evidence',errors[0]); self.assertIn('pending guidance',errors[2])

    def test_requested_followup_does_not_count_as_executed_advice_evidence(self):
        agent=self.agent([response(1),self.request(2),
            response(3,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.request(4,ids=['observation-3'],advice_progress=[progress('advice-1',None,status='in_progress')]),
            response(5,args={'command':'true','state':STATE,'advice_decision':adopt('advice-2'),
                            'advice_progress':[progress('advice-1','observation-3')]}),
            self.finish(6,advice_progress=[progress('advice-2','observation-5')])],policy='advice_requested')
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        self.assertEqual(agent.advisor_calls,2)
        self.assertEqual([a['actor_call'] for a in agent.advice_records['advice-1']['actions']],[3])
        self.assertTrue(all(r['status']=='validated' for r in agent.advice_records.values()))

    def test_requested_old_citation_uses_visible_output_not_elided_raw_output(self):
        command="printf 'old-start'; printf '%0500d' 0; printf 'old-end'"
        agent=self.agent([response(1,args={'command':command,'state':STATE}),
            *[response(i,args={'command':"printf '%0500d' 1",'state':STATE}) for i in range(2,15)],
            self.request(15),response(16,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
            self.finish(17,advice_progress=[progress('advice-1','observation-16')])],
            policy='advice_requested',max_context_chars=2000)
        self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
        packet=next(e['payload']['packet'] for e in self.events(agent) if e['kind']=='advice_requested')
        cited=packet['cited_evidence']['observation-1']
        self.assertIn('old-start',cited); self.assertIn('omitted',cited)
        self.assertNotIn('0'*200,cited)
        self.assertLessEqual(len(json.dumps(packet,ensure_ascii=False)),2000)

    def test_requested_fallback_is_one_consultation_and_failure_preserves_usage(self):
        for fail in (False,True):
            with self.subTest(fail=fail):
                advisor=self.routed_advisor(fail_both=fail)
                agent=self.agent([response(1),self.request(2),
                    response(3,args={'command':'true','state':STATE,'advice_decision':adopt('advice-1')}),
                    self.finish(4,advice_progress=[progress('advice-1','observation-3')])],
                    policy='advice_requested',max_calls=1,advisor=advisor)
                if fail:
                    with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
                    self.assertEqual(agent.n_calls,2)
                else:self.assertEqual(agent.run('Issue')['exit_status'],'Submitted')
                summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
                self.assertEqual(summary['advisor']['logical_calls'],1)
                self.assertEqual(summary['advisor']['physical_api_calls'],2)
                self.assertEqual(summary['advice_calls_remaining'],0)
                self.assertEqual(summary['advisor']['observed_cost_usd'],'0.003' if fail else '0.023')

    def test_requested_malformed_guidance_consumes_allowance_and_records_actor_cost(self):
        agent=self.agent([response(1),self.request(2)],policy='advice_requested',max_calls=1,advisor_content='{}')
        with self.assertRaises(AdvisorUnavailable):agent.run('Issue')
        summary=json.loads((agent.advice_store.run_dir/'summary.json').read_text())
        self.assertEqual(summary['advisor']['logical_calls'],1)
        self.assertEqual(summary['advisor']['cost_usd'],'0.02')
        self.assertEqual(summary['actor']['api_calls'],2)

    def test_requested_expired_wall_time_denies_before_advisor_dispatch(self):
        agent=self.agent([response(1),self.request(2)],policy='advice_requested',wall_time_limit_seconds=5)
        transport=agent.model.backend.transport
        def expired(request,timeout):
            completion=transport(request,timeout)
            if len(self.requests)==2:
                agent._start_time=time.time()-10
            return completion
        agent.model.backend.transport=expired
        self.assertEqual(agent.run('Issue')['exit_status'],'TimeExceeded')
        self.assertEqual(agent.advice_denials,{'no_time_capacity':1})
        self.assertEqual(agent.advisor_calls,0)

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
            self.assertIn('interventions.py',names)
            self.assertEqual(set(json.loads((Path(temp)/'poc-source.json').read_text())),names)


if __name__=='__main__':
    unittest.main()
