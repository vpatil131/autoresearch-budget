"""Deterministic Phase 3 progress collector; semantic labels remain actor claims."""
from __future__ import annotations

import hashlib
import json
import re


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


def normalized_check_output(text):
    """Remove known presentation noise; keep test names, assertions and values."""
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    output = []
    for line in text.splitlines():
        # Only a logging prefix, not timestamps occurring in assertion values.
        line = re.sub(r"^(\s*\[?)\d{4}-\d{2}-\d{2}[T ][0-9:.]+(?:Z|[+-]\d{2}:\d{2})?(\]?\s)",
                      r"\1<timestamp>\2", line)
        if re.match(r"\s*(?:=|\d+ (?:failed|passed|skipped)|--- (?:FAIL|PASS):|FAIL\s|ok\s|Tests run:|Time elapsed:|Time:|Duration\s|[✓×❯])", line):
            line = re.sub(r"\b(?:in|took|elapsed[:=]?|duration)\s+\d+(?:\.\d+)?\s*(?:seconds?|secs?|ms|s)\b",
                          "<duration>", line, flags=re.I)
            line = re.sub(r"\(\d+(?:\.\d+)?\s*(?:ms|s)\)", "(<duration>)", line)
            line = re.sub(r"(?<=\s)\d+(?:\.\d+)?s$", "<duration>", line)
        output.append(line)
    return '\n'.join(output).strip()


def recovery_question(text):
    value = text.strip()
    return "" if re.match(r"^(?:none|no question|no unresolved question)(?:[.!]?\s*$|\.?\s*\(|\s*:)", value, re.I) else value


class RecoveryCollector(InterventionCollector):
    """Recovery-v1 only: legacy collection remains unchanged for plan_review."""
    version = "phase3-recovery-v1"

    def __init__(self, policy):
        super().__init__(policy)
        self.last_activity_call = 0
        self.candidate_hash = None
        self.candidate_grace_after = None
        self.grace_used = False
        self.failures = {}
        self.concern = None
        self.concern_number = 0
        self.concern_history = []
        self.semantic_revision = 0
        self.semantic_outcomes = {}
        self.last_review_call = -policy.stall_steps
        self.evidence_returns = {}
        self.decisions = []

    def advance(self, call):
        # New information is progress, not proof that an open concern is solved.
        self.last_progress_call = call
        self.semantic_revision += 1
        self.grace_used = False
        self.candidate_grace_after = None

    def candidate_changed(self, source_hash, call):
        if self.candidate_hash is not None and source_hash != self.candidate_hash:
            self.last_activity_call = call
            if not self.grace_used:
                self.candidate_grace_after = call
                self.grace_used = True
        self.candidate_hash = source_hash
        self.failure_count = max((f['count'] for f in self.failures.values() if f['source_hash'] == source_hash), default=0)

    def validate_recovery(self, action):
        state = action['state']
        if state.get('concern_id') and (not self.concern or state['concern_id'] != self.concern['id']):
            raise ValueError('concern_id must reference the active recovery concern')
        if state.get('concern_status') in {'resolved', 'replace'}:
            if not self.concern or not state.get('concern_id'):
                raise ValueError('Resolve/replace requires the active concern_id')
            if not (state.get('evidence_refs') or action.get('evidence_ids')):
                raise ValueError('Resolve/replace requires visible evidence')

    def action(self, action, call):
        self.validate_recovery(action)
        state = action['state']
        if state['phase'] != 'inspect' and not self.localized:
            self.localized = True
            self.last_progress_call = call
        status = state.get('concern_status', 'open')
        if self.concern and status in {'resolved', 'replace'}:
            self.concern_history.append({**self.concern, 'status':status, 'closed_call':call,
                                         'resolution_evidence':state.get('evidence_refs') or action['evidence_ids']})
            self.concern = None
            self.high_count = self.conflict_count = 0
            self.advance(call)
        nominated = state['uncertainty'] == 'high' or state['blocker'] == 'conflicting_evidence'
        if nominated and status != 'resolved' and state['blocker'] != 'environment':
            if self.concern is None:
                self.concern_number += 1
                self.concern = {'id':f'concern-{self.concern_number}', 'status':'open',
                                'first_call':call, 'question':'', 'evidence_refs':[]}
        if self.concern and state['blocker'] != 'environment':
            question = recovery_question(state.get('uncertainty_question', '') or action.get('question', ''))
            refs = state.get('evidence_refs') or action.get('evidence_ids', [])
            if question and refs:
                self.concern.update(question=question, evidence_refs=refs, last_supported_call=call)
            supported = bool(self.concern['question'] and self.concern['evidence_refs'])
            if supported:
                self.high_count += int(state['uncertainty'] == 'high')
                self.conflict_count += int(state['blocker'] == 'conflicting_evidence')
        progress = action.get('progress')
        if progress:
            refs = progress['evidence_refs']
            facts = {(progress['kind'], self.evidence_facts[ref]) for ref in refs}
            fresh = facts - self.progress_facts
            self.progress_facts.update(facts)
            # Candidate/environment work is activity. A claimed validation must
            # cite successful completed commands; failed checks remain evidence.
            behavioral = progress['kind'] in {'localization', 'discriminating_check', 'validation'}
            if progress['kind'] == 'validation':
                behavioral = all(self.evidence_returns.get(ref) == 0 for ref in refs)
            if fresh and behavioral:
                self.advance(call)
            elif fresh:
                self.last_activity_call = call
            return bool(fresh)
        return False

    def publish_plan(self, action):
        previous = self.plan
        plan = super().publish_plan(action)
        if previous and plan['key'] != previous['key'] and self.concern:
            self.concern_history.append({**self.concern, 'status':'replaced_by_plan', 'plan_id':plan['plan_id']})
            self.concern = None
            self.high_count = self.conflict_count = 0
        return plan

    def observation(self, action, result, call, observation_id, source_hash, visible_text):
        self.last_observation = observation_id
        self.last_activity_call = call
        normalized = normalized_check_output(visible_text)
        fact = fingerprint([action['command'], source_hash, result.get('returncode'), normalized])
        self.evidence_facts[observation_id] = fact
        self.evidence_returns[observation_id] = result.get('returncode') if not result.get('exception_info') and not result.get('timed_out') else None
        if self.candidate_grace_after is not None and call > self.candidate_grace_after:
            self.candidate_grace_after = None  # At most one completed command.
        check = action.get('check')
        record = None
        if check:
            semantic = (check['kind'] == 'semantic' and result.get('returncode') is not None
                        and not result.get('exception_info') and not result.get('timed_out')
                        and action['state']['blocker'] != 'environment')
            unexpected = semantic and check['expected'] == 'success' and result['returncode'] != 0
            identity = fingerprint([check['id'], action['command'].strip(), source_hash, check['expected']])
            signature = fingerprint([result.get('returncode'), normalized])
            previous = self.failures.get(identity)
            if semantic:
                streak = previous['count'] + 1 if unexpected and previous and previous['signature'] == signature else int(unexpected)
                self.failures[identity] = {'count':streak, 'signature':signature, 'check_id':check['id'],
                                           'command':action['command'].strip(), 'source_hash':source_hash,
                                           'observation_id':observation_id}
            else:
                streak = 0
            current_hash = self.candidate_hash or source_hash
            self.failure_count = max((f['count'] for f in self.failures.values() if f['source_hash'] == current_hash), default=0)
            record = {'observation_id':observation_id, 'actor_call':call, 'source_hash':source_hash,
                      'check':check, 'command':action['command'], 'returncode':result.get('returncode'),
                      'unexpected_failure':bool(unexpected), 'failure_streak':streak,
                      'semantic_result':bool(semantic), 'failure_signature':signature, 'detector_version':self.version}
            self.checks.append(record)
            if semantic:
                behavioral_check = (check['id'], action['command'].strip(), check['expected'])
                # An edit followed by the same failing behavior is still activity.
                # Track per-check transitions independently of candidate hashes.
                if self.semantic_outcomes.get(behavioral_check) != signature:
                    self.advance(call)
                self.semantic_outcomes[behavioral_check] = signature
        if self.fallback_after is not None and call > self.fallback_after:
            self.fallback_ready = True
        return record

    def reasons(self, call, state):
        if state.get('blocker') == 'environment':
            return []
        reasons = []
        if (self.localized and call - self.last_progress_call >= self.policy.stall_steps
                and call - self.last_review_call >= self.policy.stall_steps
                and self.candidate_grace_after is None):
            reasons.append('stall')
        if self.failure_count >= self.policy.failure_threshold:
            reasons.append('repeated_failure')
        if self.concern and not (self.concern['question'] and self.concern['evidence_refs']):
            reasons.append('concern_evidence_required')
        if self.high_count >= self.policy.uncertainty_steps:
            reasons.append('persistent_uncertainty')
        if self.conflict_count >= self.policy.contradiction_steps:
            reasons.append('conflicting_evidence')
        return reasons

    def recovery_key(self, reasons, state, source_hash):
        # Incidental passing checks and paraphrasing cannot reopen reviewed
        # failures/concerns. A new candidate or new supporting facts can.
        failures = sorted((identity, failure['signature']) for identity, failure in self.failures.items()
                          if failure['source_hash'] == source_hash and failure['count'] >= self.policy.failure_threshold)
        concern_facts = sorted({self.evidence_facts.get(ref, ref) for ref in self.concern['evidence_refs']}) if self.concern else []
        return fingerprint([self.concern and self.concern['id'], self.plan and self.plan['plan_id'],
                            source_hash, self.semantic_revision if 'stall' in reasons else None,
                            failures if 'repeated_failure' in reasons else [], concern_facts, sorted(reasons)])

    def refresh_checkpoint_support(self, call, state, source_hash):
        if (self.checkpoint and 'concern_evidence_required' in self.checkpoint['reasons']
                and self.concern and self.concern['question'] and self.concern['evidence_refs']):
            reasons = list(dict.fromkeys([*[r for r in self.checkpoint['reasons'] if r != 'concern_evidence_required'],
                                          *self.reasons(call, state)]))
            self.checkpoint['reasons'] = reasons
            self.checkpoint['key'] = self.recovery_key(reasons, state, source_hash)
            return True
        return False

    def checkpoint_resolved(self):
        checkpoint = self.checkpoint
        if not checkpoint:
            return True
        reasons = checkpoint['reasons']
        resolved = []
        for reason in reasons:
            if reason == 'stall':
                resolved.append(self.last_progress_call > checkpoint['actor_call'])
            elif reason == 'repeated_failure':
                targets = {tuple(target) for target in checkpoint.get('failure_checks', [])}
                passed = {(c['check']['id'], c['command'].strip()) for c in self.checks if c['actor_call'] > checkpoint['actor_call']
                          and c['semantic_result'] and c['check']['expected'] == 'success'
                          and c['returncode'] == 0 and c['source_hash'] == self.candidate_hash}
                resolved.append(bool(targets) and targets <= passed)
            elif reason == 'concern_evidence_required':
                closed = checkpoint.get('concern_id') is not None and (
                    not self.concern or self.concern['id'] != checkpoint['concern_id'])
                resolved.append(closed or bool(self.concern and self.concern['question'] and self.concern['evidence_refs']))
            else:
                resolved.append(checkpoint.get('concern_id') is not None and
                                (not self.concern or self.concern['id'] != checkpoint['concern_id']))
        return all(resolved)

    def finish_checkpoint(self, call, *, reviewed=False, attempted=False, disposition='recorded'):
        if self.checkpoint:
            if reviewed or attempted:
                self.handled_recovery.add(self.checkpoint['key'])
            self.decisions.append({'actor_call':call, 'checkpoint':dict(self.checkpoint), 'disposition':disposition})
        self.checkpoint = None
        self.checkpoint_announced = self.fallback_ready = False
        self.fallback_after = None
        if reviewed:
            self.last_review_call = call

    def summary(self):
        return {**super().summary(), 'detector_version':self.version,
                'last_activity_call':self.last_activity_call, 'semantic_revision':self.semantic_revision,
                'concern':self.concern, 'concern_history':self.concern_history,
                'recovery_decisions':self.decisions}
