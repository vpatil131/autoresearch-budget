import unittest

from autoresearch_budget.advisor import AdvicePolicy
from autoresearch_budget.contracts import actor_tools, validate_action, validate_advice_context
from autoresearch_budget.interventions import InterventionCollector

STATE = {'phase': 'diagnose', 'uncertainty': 'low', 'blocker': 'none'}


class InterventionsTest(unittest.TestCase):
    def collector(self):
        return InterventionCollector(AdvicePolicy.load({'policy': 'hybrid'}))

    def action(self, **extra):
        return {'name': 'execute', 'command': 'run-test', 'state': STATE,
                'check': {'id': 'test-1', 'kind': 'semantic', 'expected': 'success'}, **extra}

    def observe(self, collector, action=None, *, call=1, source='source', result=None):
        action = action or self.action()
        collector.action(action, call)
        return collector.observation(action, result or {'returncode': 1}, call,
                                     f'observation-{call}', source, 'failure')

    def test_failure_identity_includes_command_source_and_check(self):
        collector = self.collector()
        for call in range(1, 4): self.observe(collector, call=call)
        self.assertEqual(collector.reasons(3, STATE), ['repeated_failure'])
        self.observe(collector, call=4, source='different-test-or-source')
        self.assertEqual(collector.failure_count, 1)
        self.observe(collector, self.action(command='different-test'), call=5)
        self.assertEqual(collector.failure_count, 1)
        self.observe(collector, call=6, result={'returncode': 0})
        self.assertEqual(collector.failure_count, 0)

    def test_expected_failure_environment_unknown_and_timeout_are_not_semantic_failures(self):
        for kind, expected, result in (
            ('semantic', 'failure', {'returncode': 1}),
            ('environment', 'success', {'returncode': 1}),
            ('unknown', 'success', {'returncode': 1}),
            ('semantic', 'success', {'returncode': -1, 'timed_out': True}),
            ('semantic', 'success', {'returncode': -1, 'exception_info': 'environment failure'}),
        ):
            with self.subTest(kind=kind, expected=expected, result=result):
                collector = self.collector()
                action = self.action(check={'id': 'test-1', 'kind': kind, 'expected': expected})
                for call in range(1, 4): self.observe(collector, action, call=call, result=result)
                self.assertEqual(collector.failure_count, 0)
                self.assertNotIn('repeated_failure', collector.reasons(3, STATE))

    def test_new_observation_ids_do_not_establish_new_progress(self):
        collector = self.collector()
        for call in range(1, 10): self.observe(collector, call=call)
        self.assertEqual(collector.last_progress_call, 1)
        self.assertIn('stall', collector.reasons(9, STATE))
        self.assertEqual(len(set(collector.evidence_facts.values())), 1)

    def test_initial_inspection_is_not_a_stall(self):
        collector = self.collector()
        collector.action({'state': {**STATE, 'phase': 'inspect'}}, 20)
        self.assertEqual(collector.reasons(20, STATE), [])
        self.assertEqual(collector.reasons(20, {**STATE, 'blocker': 'environment'}), [])

    def test_unsupported_uncertainty_does_not_trigger(self):
        collector = self.collector()
        for call in range(1, 5):
            collector.action({'state': {**STATE, 'uncertainty': 'high', 'blocker': 'conflicting_evidence'}}, call)
        self.assertEqual(collector.reasons(4, STATE), [])

    def test_contradictions_require_repeated_question_with_evidence(self):
        collector = self.collector()
        state = {**STATE, 'blocker': 'conflicting_evidence', 'uncertainty_question': 'Which result fits?',
                 'evidence_refs': ['observation-1']}
        for call in (1, 2): collector.action({'state': state}, call)
        self.assertEqual(collector.reasons(2, state), ['conflicting_evidence'])
        collector.action({'state': {**state, 'uncertainty_question': 'Different question'}}, 3)
        self.assertEqual(collector.reasons(3, state), [])

    def test_phase3_tools_and_fields_are_disabled_for_old_contracts(self):
        self.assertNotIn('propose_plan', {t['function']['name'] for t in actor_tools(advice_enabled=True)})
        with self.assertRaises(ValueError): validate_action('execute', {'command': 'true', 'state': STATE, 'check': {'id': 't', 'kind': 'semantic', 'expected': 'success'}})
        valid = validate_action('execute', {'command': 'true', 'state': STATE,
                               'check': {'id': 't', 'kind': 'semantic', 'expected': 'success'}}, phase3_enabled=True)
        self.assertEqual(valid['check']['id'], 't')
        with self.assertRaises(ValueError): validate_action('execute', {**valid, 'check': {'id': 't', 'kind': 'invalid', 'expected': 'success'}}, phase3_enabled=True)

    def test_progress_and_uncertainty_references_must_be_visible(self):
        for extras in ({'state': {**STATE, 'evidence_refs': ['secret-observation']}},
                       {'progress': {'kind': 'localization', 'summary': 'Found caller', 'evidence_refs': ['secret-patch']}}):
            action = self.action(**extras)
            validate_action('execute', {k: v for k, v in action.items() if k != 'name'}, phase3_enabled=True)
            with self.assertRaises(ValueError):
                validate_advice_context(action, pending_advice=None, records={}, observation_ids={'observation-1'})

    def test_phase3_policy_settings_validate_and_never_use_fixed_intervals(self):
        for mode in ('phase3_control', 'plan_review', 'recovery_review', 'hybrid'):
            policy = AdvicePolicy.load({'policy': mode})
            self.assertTrue(policy.actor_requested)
            self.assertEqual(policy.reviews_plans, mode in {'plan_review', 'hybrid'})
            self.assertEqual(policy.reviews_recovery, mode in {'recovery_review', 'hybrid'})
            self.assertFalse(policy.scheduled(10, 80, -1))
        for field in ('stall_steps', 'failure_threshold', 'uncertainty_steps', 'contradiction_steps', 'min_actor_calls_remaining'):
            for value in (True, 0, '8'):
                with self.assertRaises(ValueError): AdvicePolicy.load({'policy': 'hybrid', field: value})


if __name__ == '__main__': unittest.main()
