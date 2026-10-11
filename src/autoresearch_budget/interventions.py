"""Deterministic Phase 3 progress collector; semantic labels remain actor claims."""
from __future__ import annotations

import hashlib
import json


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class InterventionCollector:
    def __init__(self, policy):
        self.policy = policy
        self.localized = False
        self.last_progress_call = 0
        self.progress_facts = set()
        self.evidence_facts = {}
        self.checks = []
        self.failure_key = None
        self.failure_count = 0
        self.concern_key = None
        self.high_count = self.conflict_count = 0
        self.plan = None
        self.plan_number = 0
        self.reviewed_plans = set()
        self.handled_recovery = set()
        self.checkpoint = None
        self.checkpoint_announced = False
        self.fallback_ready = False
        self.fallback_after = None
        self.last_observation = None
        self.inspection_checkpoint = None
        self.last_inspection_response = 0
        self.inspection_responses = []
        self.plans = []

    def action(self, action, call):
        state = action["state"]
        if state["phase"] != "inspect" and not self.localized:
            self.localized = True
            self.last_progress_call = call
        concern = state.get("uncertainty_question", "").strip()
        refs = state.get("evidence_refs", [])
        key = fingerprint([concern, state.get("hypothesis", self.plan and self.plan["hypothesis"])])
        if key != self.concern_key:
            self.high_count = self.conflict_count = 0
        self.concern_key = key
        supported = bool(concern and refs and state["blocker"] != "environment")
        self.high_count = self.high_count + 1 if supported and state["uncertainty"] == "high" else 0
        self.conflict_count = self.conflict_count + 1 if supported and state["blocker"] == "conflicting_evidence" else 0
        progress = action.get("progress")
        if progress:
            # IDs alone cannot create progress: compare the underlying visible facts.
            facts = {(progress["kind"], self.evidence_facts[ref]) for ref in progress["evidence_refs"]}
            fresh = facts - self.progress_facts
            self.progress_facts.update(facts)
            if fresh:
                self.advance(call)
            return bool(fresh)
        return False

    def advance(self, call):
        self.last_progress_call = call
        self.high_count = self.conflict_count = 0
        self.checkpoint = None
        self.checkpoint_announced = False
        self.fallback_ready = False
        self.fallback_after = None

    def plan_key(self, action):
        fields = ("hypothesis", "approach", "falsifying_check", "validation_plan")
        return fingerprint([" ".join(action[field].split()) for field in fields])

    def validate_plan(self, action):
        previous = action["previous_plan_id"]
        if self.plan is None:
            if previous:
                raise ValueError("Initial plan requires previous_plan_id='' (no prior plan)")
        elif self.plan_key(action) != self.plan["key"]:
            if previous != self.plan["plan_id"]:
                raise ValueError(f"Revision must reference current plan {self.plan['plan_id']} in previous_plan_id")
        elif previous not in {self.plan["plan_id"], self.plan["previous_plan_id"]}:
            raise ValueError("Duplicate plan must reference its recorded predecessor or current plan ID")

    def publish_plan(self, action):
        self.validate_plan(action)
        fields = ("hypothesis", "approach", "falsifying_check", "validation_plan")
        key = self.plan_key(action)
        if self.plan is None or self.plan["key"] != key:
            self.plan_number += 1
            self.plan = {"plan_id": f"plan-{self.plan_number}", "key": key,
                         **{field: action[field] for field in fields}, "evidence_ids": action["evidence_ids"],
                         "previous_plan_id": action["previous_plan_id"], "change_reason": action["change_reason"]}
            self.plans.append(self.plan)
        self.inspection_checkpoint = None
        return self.plan

    def inspection_due(self, call):
        return (self.plan is None and self.last_observation is not None
                and call - self.last_inspection_response >= self.policy.inspection_steps)

    def record_inspection(self, action, call):
        self.inspection_responses.append({"actor_call": call, "inspection": action["inspection"],
                                          "question": action["question"], "next_check": action["next_check"],
                                          "evidence_ids": action["evidence_ids"]})
        self.last_inspection_response = call
        self.inspection_checkpoint = None

    def observation(self, action, result, call, observation_id, source_hash, visible_text):
        self.last_observation = observation_id
        fact = fingerprint([action["command"], source_hash, result.get("returncode"), visible_text])
        self.evidence_facts[observation_id] = fact
        check = action.get("check")
        if check:
            unexpected = (check["kind"] == "semantic" and check["expected"] == "success"
                          and result.get("returncode") not in (None, 0)
                          and not result.get("exception_info") and not result.get("timed_out")
                          and action["state"]["blocker"] != "environment")
            # Exact command, declared check identity, and full source/test diff are
            # conservative identities. We do not infer equivalent checks from prose.
            key = fingerprint([check["id"], action["command"].strip(), source_hash,
                               result.get("returncode"), visible_text])
            self.failure_count = self.failure_count + 1 if unexpected and key == self.failure_key else int(unexpected)
            self.failure_key = key if unexpected else None
            record = {"observation_id": observation_id, "actor_call": call,
                      "source_hash": source_hash, "check": check, "command": action["command"],
                      "returncode": result.get("returncode"), "unexpected_failure": unexpected,
                      "failure_streak": self.failure_count}
            self.checks.append(record)
            if (check["kind"] == "semantic" and not result.get("exception_info")
                    and not result.get("timed_out") and action["state"]["blocker"] != "environment"):
                progress_key = ("check", fact)
                if progress_key not in self.progress_facts:
                    self.progress_facts.add(progress_key)
                    self.advance(call)
        if self.fallback_after is not None and call > self.fallback_after:
            self.fallback_ready = True
        return self.checks[-1] if check else None

    def reasons(self, call, state):
        reasons = []
        if self.localized and call - self.last_progress_call >= self.policy.stall_steps:
            reasons.append("stall")
        if self.failure_count >= self.policy.failure_threshold:
            reasons.append("repeated_failure")
        if self.high_count >= self.policy.uncertainty_steps:
            reasons.append("persistent_uncertainty")
        if self.conflict_count >= self.policy.contradiction_steps:
            reasons.append("conflicting_evidence")
        if state.get("blocker") == "environment":
            return []
        return reasons

    def recovery_key(self, reasons, state, source_hash):
        # Same unresolved concern on the same candidate is reviewed once. New
        # visible evidence or a changed candidate permits another intervention.
        return fingerprint([state.get("uncertainty_question") or state.get("hypothesis"),
                            source_hash, self.evidence_facts.get(self.last_observation)])

    def summary(self):
        return {"plan": self.plan, "plan_versions": self.plan_number, "plans": self.plans,
                "inspection_checkpoint": self.inspection_checkpoint, "inspection_responses": self.inspection_responses,
                "last_progress_call": self.last_progress_call, "failure_streak": self.failure_count,
                "checks": self.checks, "checkpoint": self.checkpoint,
                "reviewed_plans": len(self.reviewed_plans), "handled_recovery": len(self.handled_recovery)}
