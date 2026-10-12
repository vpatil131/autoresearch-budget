"""Shared pre-action advice requirements and unchanged rejection state."""
import unittest

from autoresearch_budget.contracts import advice_turn_requirements, actor_schema_correction


class AdviceMessagingTest(unittest.TestCase):
    def test_no_advice_has_no_reminder(self):
        self.assertEqual(advice_turn_requirements(None, {}), "")
        self.assertEqual(advice_turn_requirements(None, {"old": {"status": "validated"}}), "")

    def test_first_acknowledgment_requires_decision_and_omits_progress(self):
        text = advice_turn_requirements("advice-1", {"advice-1": {"status": "delivered"}})
        self.assertIn('Pending acknowledgment ID: "advice-1"', text)
        self.assertIn("Include advice_decision", text)
        self.assertIn("Progress IDs required this turn: []", text)
        self.assertIn("Omit advice_progress", text)

    def test_pending_followup_is_excluded_from_prior_progress(self):
        records = {"advice-2": {"status": "in_progress"}, "advice-1": {"status": "acknowledged"},
                   "advice-3": {"status": "delivered"}, "advice-4": {"status": "rejected"}}
        text = advice_turn_requirements("advice-3", records)
        self.assertIn('Progress IDs required this turn: ["advice-1", "advice-2"]', text)
        self.assertIn('Do not include "advice-3" in advice_progress this turn', text)
        self.assertIn("after successful acknowledgment", text)

    def test_after_acknowledgment_new_id_joins_progress(self):
        text = advice_turn_requirements(None, {"advice-3": {"status": "acknowledged"}})
        self.assertIn("Omit advice_decision", text)
        self.assertIn('Progress IDs required this turn: ["advice-3"]', text)

    def test_correction_explains_atomic_rejection_and_retains_requirements(self):
        records = {"advice-1": {"status": "in_progress"}, "advice-2": {"status": "delivered"}}
        text = actor_schema_correction("wrong progress IDs", pending_advice="advice-2", records=records)
        self.assertIn("Schema error: wrong progress IDs", text)
        self.assertIn("not executed or accepted", text)
        self.assertIn("decision and progress were not recorded", text)
        self.assertIn(advice_turn_requirements("advice-2", records), text)
        self.assertEqual(records["advice-2"]["status"], "delivered")
