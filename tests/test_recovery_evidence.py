"""Offline regressions for masked execution and false semantic evidence."""
import unittest

from autoresearch_budget.recovery_evidence import direct_check_command, recovery_check_evidence


class RecoveryEvidenceTest(unittest.TestCase):
    def evidence(self, command="npx vitest run regression.spec.ts", output="1 passed", **result):
        return recovery_check_evidence(
            {"command": command, "state": {"blocker": "none"},
             "check": {"kind": "semantic", "expected": "success", "id": "regression"}},
            {"output": output, "returncode": 0, "exception_info": "", **result},
            candidate_changed=False)

    def test_direct_checks_preserve_success_and_failed_behavioral_outcomes(self):
        for command in ("npx vitest run test.ts", "cd /core && npx vitest run test.ts",
                        "cd -- /core && python -m pytest test.py", "CI=1 npx vitest run test.ts",
                        "python -c 'print(1); raise SystemExit(1)'", "printf 'a|b'"):
            with self.subTest(command=command):
                self.assertTrue(direct_check_command(command))
                self.assertTrue(self.evidence(command, "AssertionError: expected a got ac", returncode=1)["eligible"])

    def test_shell_masking_and_wrappers_do_not_qualify(self):
        for command in ("false | tail -20", "false; echo done", "false || true",
                        "cd /core && false && true", "bash -o pipefail -c 'false | tail -20'",
                        "command sh -c 'false; true'", "env sh -c 'false; true'",
                        "false > /tmp/test.log", "false\necho done", "echo $(false)",
                        "echo `false`", "timeout 300 npx vitest run test.ts"):
            with self.subTest(command=command):
                e = self.evidence(command)
                self.assertFalse(e["eligible"])
                self.assertEqual(e["reason"], "ambiguous_check_exit_status")
                self.assertFalse(e["exit_status_reliable"])

    def test_original_vue_module_error_is_environment_even_with_zero_exit(self):
        e = self.evidence("node /tmp/repro.mjs | tail -20",
                          "Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'puppeteer' imported from /tmp/repro.mjs")
        self.assertEqual(e["reason"], "observed_environment_failure")
        self.assertFalse(e["eligible"])

    def test_direct_setup_failure_is_not_a_behavioral_failure(self):
        for text in ("ModuleNotFoundError: No module named 'pytest'",
                     "Error: Cannot find module 'vitest'", "/bin/sh: 1: npx: not found",
                     "/bin/sh: 1: Bad substitution", "No test files found, exiting with code 1",
                     "Failed to launch the browser process"):
            with self.subTest(text=text):
                e = self.evidence(output=text, returncode=1)
                self.assertFalse(e["eligible"])
                self.assertTrue(e["exit_status_reliable"])
                self.assertEqual(e["reason"], "observed_environment_failure")

    def test_incomplete_or_timed_out_execution_does_not_qualify(self):
        for values in ({"timed_out": True}, {"exception_info": "execution error"}, {"returncode": None}):
            with self.subTest(values=values):
                self.assertEqual(self.evidence(**values)["reason"], "execution_incomplete")

    def test_edit_and_check_are_not_evidence_for_the_new_candidate(self):
        e = recovery_check_evidence({"command": "npx vitest run test.ts", "state": {},
            "check": {"kind": "semantic"}}, {"output": "passed", "returncode": 0}, candidate_changed=True)
        self.assertEqual(e["reason"], "candidate_changed_during_check")
