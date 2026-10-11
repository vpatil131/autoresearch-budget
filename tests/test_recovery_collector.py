"""Recovery-v1 behavioral cases and pre-change plan-policy compatibility hashes."""
import hashlib
import inspect
import json
import unittest

from autoresearch_budget.advisor import AdvicePolicy
from autoresearch_budget.contracts import actor_prompts, actor_tools, validate_action, validate_advice_context
from autoresearch_budget.interventions import InterventionCollector, RecoveryCollector, normalized_check_output, recovery_question

STATE = {'phase':'diagnose', 'uncertainty':'low', 'blocker':'none'}


class RecoveryCollectorTest(unittest.TestCase):
    def setUp(self):
        self.c = RecoveryCollector(AdvicePolicy.load({'policy':'recovery_review'}))
        self.c.candidate_changed('source', 0)

    def action(self, **extra):
        return {'name':'execute', 'command':'test regression', 'state':STATE,
                'check':{'id':'regression','kind':'semantic','expected':'success'}, **extra}

    def observe(self, call, *, action=None, output='AssertionError: expected new, got old', rc=1, source='source', **result):
        a = action or self.action()
        self.c.action(a, call)
        return self.c.observation(a, {'returncode':rc, **result}, call, f'observation-{call}', source, output)

    def test_durations_and_timestamps_do_not_hide_repeated_failure(self):
        for call in range(1,4):
            record = self.observe(call, output=f'2026-10-11T10:01:0{call}Z FAILED test_a\nAssertionError: expected 2 got 1\n1 failed in {call}.12 seconds')
        self.assertEqual(record['failure_streak'], 3)
        self.assertEqual(self.c.last_progress_call, 1)
        self.assertIn('repeated_failure', self.c.reasons(3, STATE))

    def test_different_assertion_values_are_not_collapsed(self):
        self.observe(1, output='AssertionError: expected 2 got 1')
        record = self.observe(2, output='AssertionError: expected 2 got 3')
        self.assertEqual(record['failure_streak'], 1)
        self.assertEqual(self.c.last_progress_call, 2)

    def test_unrelated_passing_check_does_not_erase_failure_history(self):
        self.observe(1); self.observe(2)
        self.observe(3, action=self.action(command='test other', check={'id':'other','kind':'semantic','expected':'success'}), rc=0)
        self.assertEqual(self.observe(4)['failure_streak'], 3)
        self.observe(5, rc=0)
        self.assertEqual(self.c.failure_count, 0)

    def test_unknown_inspection_does_not_break_streak(self):
        self.observe(1); self.observe(2)
        self.observe(3, action={'name':'execute','command':'read file','state':STATE}, rc=0)
        self.assertEqual(self.observe(4)['failure_streak'], 3)

    def test_new_candidate_does_not_inherit_failure_streak(self):
        self.observe(1); self.observe(2); self.observe(3)
        self.c.candidate_changed('new', 4)
        self.assertEqual(self.c.failure_count, 0)
        self.assertEqual(self.observe(5, source='new')['failure_streak'], 1)

    def test_edit_then_same_failing_behavior_does_not_reset_stall_clock(self):
        self.observe(1)
        self.c.candidate_changed('edit-a',8)
        self.observe(9,source='edit-a')
        self.assertEqual(self.c.last_progress_call,1)
        self.assertIn('stall',self.c.reasons(9,STATE))
        self.c.candidate_changed('edit-b',10)
        self.assertIn('stall',self.c.reasons(10,STATE))
        self.observe(11,source='edit-b',rc=0,output='Passed')
        self.assertEqual(self.c.last_progress_call,11)

    def test_expected_failures_and_environment_are_not_failed_fixes(self):
        for kind, expected, extra in [('semantic','failure',{}),('environment','success',{}),
                                     ('unknown','success',{}),('semantic','success',{'timed_out':True})]:
            with self.subTest(kind=kind,expected=expected,extra=extra):
                self.setUp()
                for call in range(1,4):
                    self.observe(call, action=self.action(check={'id':'x','kind':kind,'expected':expected}), **extra)
                self.assertEqual(self.c.failure_count,0)

    def test_one_candidate_experiment_and_no_indefinite_edit_grace(self):
        self.c.action({'state':STATE},1)
        self.assertIn('stall', self.c.reasons(9,STATE))
        self.c.candidate_changed('edit-a',9)
        self.assertNotIn('stall',self.c.reasons(9,STATE))
        self.c.candidate_changed('edit-b',10)
        self.c.observation({'command':'edit','state':STATE}, {'returncode':0},10,'observation-10','edit-a','edited')
        self.assertIn('stall',self.c.reasons(10,STATE))

    def test_candidate_claim_is_activity_not_behavioral_progress(self):
        self.c.action({'state':STATE},1)
        self.c.evidence_facts['patch-2']='new-source'
        self.c.action({'state':STATE,'progress':{'kind':'candidate','summary':'Edited','evidence_refs':['patch-2']}},5)
        self.assertEqual(self.c.last_progress_call,1)
        self.assertEqual(self.c.last_activity_call,5)

    def test_failed_validation_claim_does_not_advance_progress(self):
        self.observe(1)
        self.c.action({'state':STATE,'progress':{'kind':'validation','summary':'Reran','evidence_refs':['observation-1']}},5)
        self.assertEqual(self.c.last_progress_call,1)

    def test_paraphrased_supported_contradiction_retains_concern(self):
        for call,question in [(1,'Why is the cache stale?'),(2,'Why does invalidation retain the old value?')]:
            state={**STATE,'blocker':'conflicting_evidence','uncertainty_question':question,'evidence_refs':['observation-0'],
                   'hypothesis':f'Description at step {call}'}
            self.c.action({'state':state},call)
        self.assertEqual(self.c.concern['id'],'concern-1')
        self.assertEqual(self.c.conflict_count,2)
        self.assertIn('conflicting_evidence',self.c.reasons(2,state))

    def test_missing_support_nominates_local_checkpoint(self):
        state={**STATE,'uncertainty':'high'}
        self.c.action({'state':state},1)
        self.assertEqual(self.c.reasons(1,state),['concern_evidence_required'])
        self.assertEqual(self.c.high_count,0)
        self.c.action({'name':'report_checkpoint','state':state,'question':'Which cause fits?',
                       'evidence_ids':['observation-1']},2)
        self.assertEqual(self.c.high_count,1)
        self.assertNotIn('concern_evidence_required',self.c.reasons(2,state))

    def test_progress_and_review_do_not_close_concern(self):
        state={**STATE,'uncertainty':'high','uncertainty_question':'Which cause?', 'evidence_refs':['observation-0']}
        self.c.action({'state':state},1)
        self.observe(2,rc=0)
        self.c.checkpoint={'key':'key','reasons':['persistent_uncertainty'],'actor_call':1,'concern_id':'concern-1'}
        self.c.finish_checkpoint(3,reviewed=True)
        self.assertEqual(self.c.concern['id'],'concern-1')
        self.assertEqual(self.c.high_count,1)
        self.assertEqual(self.c.last_progress_call,2)

    def test_explicit_resolution_requires_existing_concern_and_evidence(self):
        self.c.action({'state':{**STATE,'uncertainty':'high'}},1)
        state={**STATE,'concern_id':'concern-1','concern_status':'resolved'}
        with self.assertRaises(ValueError):self.c.action({'state':state},2)
        self.c.action({'state':{**state,'evidence_refs':['observation-1']}},2)
        self.assertIsNone(self.c.concern)
        self.assertEqual(self.c.concern_history[-1]['status'],'resolved')
        with self.assertRaises(ValueError):self.c.action({'state':state},3)

    def test_unknown_resolution_evidence_rejected_by_contract(self):
        a=self.action(state={**STATE,'concern_id':'concern-1','concern_status':'resolved','evidence_refs':['hidden']})
        validate_action('execute',{k:v for k,v in a.items() if k!='name'},phase3_enabled=True,recovery_enabled=True)
        with self.assertRaises(ValueError):validate_advice_context(a,pending_advice=None,records={},observation_ids={'observation-1'})

    def test_supported_resolution_can_close_a_local_nomination(self):
        self.c.action({'state':{**STATE,'uncertainty':'high'}},1)
        self.c.checkpoint={'key':'key','reasons':['concern_evidence_required'],'actor_call':1,'concern_id':'concern-1'}
        self.c.action({'state':{**STATE,'concern_id':'concern-1','concern_status':'resolved',
                               'evidence_refs':['observation-1']}},2)
        self.assertTrue(self.c.checkpoint_resolved())

    def test_unrelated_pass_does_not_resolve_deferred_failure_checkpoint(self):
        for call in range(1,4):self.observe(call)
        self.c.checkpoint={'key':'key','reasons':['repeated_failure'],'actor_call':3,
                           'failure_checks':[('regression','test regression')]}
        self.observe(4,rc=0,action=self.action(check={'id':'other','kind':'semantic','expected':'success'}))
        self.assertFalse(self.c.checkpoint_resolved())
        self.observe(5,rc=0)
        self.assertTrue(self.c.checkpoint_resolved())

    def test_same_check_id_with_different_command_cannot_resolve_failure(self):
        for call in range(1,4):self.observe(call)
        self.c.checkpoint={'key':'key','reasons':['repeated_failure'],'actor_call':3,
                           'failure_checks':[('regression','test regression')]}
        self.observe(4,rc=0,action=self.action(command='test unrelated'))
        self.assertFalse(self.c.checkpoint_resolved())

    def test_incidental_pass_does_not_reopen_reviewed_failure(self):
        for call in range(1,4):self.observe(call)
        reasons=['repeated_failure']
        key=self.c.recovery_key(reasons,STATE,'source')
        self.c.checkpoint={'key':key,'reasons':reasons,'actor_call':3}
        self.c.finish_checkpoint(4,reviewed=True)
        self.observe(5,rc=0,action=self.action(command='test other'))
        self.assertEqual(self.c.recovery_key(reasons,STATE,'source'),key)
        self.assertIn(key,self.c.handled_recovery)

    def test_unavailable_review_is_deduplicated_without_clearing_failure(self):
        for call in range(1,4):self.observe(call)
        self.c.checkpoint={'key':'key','reasons':['repeated_failure'],'actor_call':3}
        self.c.finish_checkpoint(4,attempted=True,disposition='review_unavailable')
        self.assertIn('key',self.c.handled_recovery)
        self.assertEqual(self.c.failure_count,3)

    def test_new_support_unblocks_mixed_stall_and_nomination(self):
        state={**STATE,'uncertainty':'high'}
        self.c.action({'state':state},1)
        self.c.checkpoint={'key':'key','reasons':['stall','concern_evidence_required'],'actor_call':9}
        self.c.action({'state':state,'question':'Which cause fits?', 'evidence_ids':['observation-1']},10)
        self.assertTrue(self.c.refresh_checkpoint_support(10,state,'source'))
        self.assertEqual(self.c.checkpoint['reasons'],['stall'])
        self.assertFalse(self.c.checkpoint_resolved())

    def test_normalization_keeps_meaningful_numeric_assertions(self):
        self.assertEqual(normalized_check_output('expected 123 got 456\n1 failed in 0.42s'),
                         'expected 123 got 456\n1 failed <duration>')
        self.assertEqual(recovery_question('None (rerun the check first)'), '')
        self.assertEqual(recovery_question('No unresolved question.'), '')
        question='None of the approaches explains this failure; why?'
        self.assertEqual(recovery_question(question), question)

    def test_normalization_preserves_time_assertions_and_go_noise(self):
        assertion = 'AssertionError: expected 2026-10-11T12:00:00Z elapsed: 5 seconds'
        self.assertEqual(normalized_check_output(assertion),assertion)
        self.assertEqual(normalized_check_output('--- FAIL: TestA (0.11s)\nFAIL module/x 0.222s'),
                         normalized_check_output('--- FAIL: TestA (0.33s)\nFAIL module/x 0.444s'))


class LegacyRecoveryIsolationTest(unittest.TestCase):
    def digest(self,value):
        return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

    def test_legacy_prompts_match_prechange_hashes(self):
        golden={'plan_review':'17b81717b71e7bc6f958e15f044a3d38575e28683015723337478bd48d2fa7cc',
                'phase3_control':'8c5f276c78236e7791d73e49bc30cfb019f3826d5872be3bda79ad10fe8411df',
                'advice_requested':'2259087609ff3f8552cd46b6303df08626aee17fa8bc1876f316591066f169aa',
                'fixed_interval':'04b90a87b21a9c08128325f450e0795786c562c7cfb9c094eba690a7cafb90fa'}
        for policy,expected in golden.items():
            self.assertEqual(self.digest(actor_prompts(policy,10,7)),expected,policy)

    def test_legacy_tools_match_prechange_hashes(self):
        self.assertEqual(self.digest(actor_tools(advice_enabled=True,phase3_enabled=True)),
                         '1f346fc0ada718b9b5752f6f5560baf4e299130a4395a12494b8ff68977c63bc')
        self.assertEqual(self.digest(actor_tools('advice-1',('advice-1',),advice_enabled=True,phase3_enabled=True)),
                         '40857e609aafa92c165813bbf48902752fa25d3194f83b95a6069b756886f105')

    def test_legacy_collector_is_unchanged(self):
        self.assertEqual(hashlib.sha256(inspect.getsource(InterventionCollector).encode()).hexdigest(),
                         'a8c13770729a8a75d279dd9a522ad67c5622d3c1d6898e6ae4c820b1d0bd05d4')

    def test_concern_fields_are_exclusive_to_recovery_policies(self):
        args={'command':'true','state':{**STATE,'concern_id':'concern-1'}}
        with self.assertRaises(ValueError):validate_action('execute',args,phase3_enabled=True)
        validate_action('execute',args,phase3_enabled=True,recovery_enabled=True)
        tools=actor_tools(phase3_enabled=True,recovery_enabled=True)
        self.assertIn('concern_id',tools[0]['function']['parameters']['properties']['state']['properties'])
