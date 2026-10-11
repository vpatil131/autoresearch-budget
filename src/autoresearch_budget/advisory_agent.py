"""mini-SWE 2.4.6 controller for scheduled, requested, and hybrid advice."""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import time
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from minisweagent.agents.default import DefaultAgent
from minisweagent.exceptions import FormatError, Submitted

from .advisor import AdvicePolicy, Advisor, AdvisorUnavailable, SCHEDULED_QUESTION, actor_usage, advice_packet, amount
from .contracts import CONTRACT_VERSION, PHASE2_CONTRACT_VERSION, PHASE3_CONTRACT_VERSION, RECOVERY_CONTRACT_VERSION, actor_prompts, OPEN_ADVICE_STATUSES, validate_action, validate_advice_context, validate_guidance
from .models import ModelError, RequestOutcomeUnknown
from .interventions import InterventionCollector, RecoveryCollector, recovery_question
from .store import RunStore

# Include new, untracked files as well as tracked/staged changes. Git binary patches
# preserve candidate contents; no hidden verifier files are queried by the controller.
PATCH_COMMAND = "bash -c " + shlex.quote("""set -o pipefail
git diff --no-ext-diff --no-color --binary HEAD &&
git ls-files --others --exclude-standard -z | while IFS= read -r -d '' file; do
  git diff --no-ext-diff --no-color --binary --no-index -- /dev/null "$file"
  rc=$?; if [ "$rc" -gt 1 ]; then exit "$rc"; fi
done""")


class AdvisoryAgent(DefaultAgent):
    def __init__(self, model, env, *, advice: dict, advisor=None, max_observation_chars=20000, **kwargs):
        self.policy = AdvicePolicy.load(advice)
        self.advisor_profile = dict(advice.get("advisor", {}))
        self.advisor = advisor
        if not isinstance(max_observation_chars, int) or max_observation_chars < 100:
            raise ValueError("max_observation_chars must be at least 100")
        self.max_observation_chars = max_observation_chars
        # The builtin mini prompt assumes bash and a shell submission sentinel.
        # Replace both prompt templates so they agree with the native tools.
        kwargs["system_template"], kwargs["instance_template"] = actor_prompts(
            self.policy.policy, self.policy.every_steps, self.policy.max_calls)
        super().__init__(model, env, **kwargs)
        if self.config.cost_limit != 0:
            raise ValueError("The advisory POC requires cost_limit=0; dollar enforcement is deferred")
        output_dir = self.config.output_path.parent if self.config.output_path else Path(".")
        secrets = tuple(value for key, value in os.environ.items()
                        if any(word in key.upper() for word in ("API_KEY", "TOKEN", "PASSWORD", "OPENROUTER_KEY")))
        self.advice_store = RunStore(output_dir / "advice", run_id=advice.get("run_id") or str(uuid4()), secrets=secrets)
        self.last_checkpoint = -1
        self.advisor_calls = self.advisor_failures = self.unknown_advisor_costs = 0
        self.advisor_api_calls = self.recorded_advisor_requests = self.advisor_fallbacks = 0
        self.advisor_cost = Decimal("0")
        self.advisor_prompt_tokens = self.advisor_completion_tokens = 0
        self.pending_advice = None
        self.advice_records: dict[str, dict] = {}
        self.state = {}
        collector_type = RecoveryCollector if self.policy.reviews_recovery else InterventionCollector
        self.interventions = collector_type(self.policy) if self.policy.phase3 else None
        self.recovery_delivered = False
        self.observation_ids: set[str] = set()
        self.visible_evidence: dict[str, str] = {}
        self.visible_patch_ids: set[str] = set()
        self.patch_evidence: dict[str, str] = {}
        self.advice_requests = 0
        self.advice_denials: dict[str, int] = {}
        self.patch_id = None
        self.patch_text = ""
        self.patch_hash = None
        self.patch_number = 0
        self.finished = False
        self.started = time.monotonic()
        if hasattr(model, "bind"):
            model.bind(self.advice_store)
        if hasattr(env, "bind"):
            env.bind(self.advice_store, lambda: self.n_calls)
        if hasattr(env, "set_time_remaining"):
            env.set_time_remaining(self._command_time_remaining)
        self.advice_store.write_metadata({
            "contract_version": RECOVERY_CONTRACT_VERSION if self.policy.reviews_recovery else (PHASE3_CONTRACT_VERSION if self.policy.phase3 else (PHASE2_CONTRACT_VERSION if self.policy.policy == "advice_requested" else CONTRACT_VERSION)),
            "policy": {key: value for key, value in self.policy.__dict__.items()
                       if key != "every_steps" or self.policy.policy == "fixed_interval"},
            "advisor": self.advisor_profile, "budget_enforced": False,
            "hidden_verifier_feedback": False, "agent_config": self.config.model_dump(mode="json"),
            "model_config": model.serialize(), "environment_config": env.serialize(),
            "max_observation_chars": max_observation_chars,
        })

    def _command_time_remaining(self):
        if not self.config.wall_time_limit_seconds:
            return None
        return self.config.wall_time_limit_seconds - (time.time() - self._start_time)

    def add_messages(self, *messages):
        for message in messages:
            self.advice_store.append_event("model_visible_message", {"actor_call": self.n_calls, "message": message})
        return super().add_messages(*messages)

    def handle_uncaught_exception(self, error):
        messages = super().handle_uncaught_exception(error)
        if isinstance(error, ModelError) and not isinstance(error, AdvisorUnavailable) and error.response:
            # Failed provider generations may still return billed usage.
            messages[0]["extra"]["response"] = error.response
            self.advice_store.append_event("actor_failed_usage", {"actor_call": self.n_calls,
                "response": error.response})
        return messages

    def run(self, task="", **kwargs):
        if self.n_calls or self.finished:
            raise ValueError("Create a new controller for each attempt")
        self.advice_store.append_event("run_started", {"actor_call": 0, "issue": task})
        identity = self.env.execute({"command": "git rev-parse --show-toplevel && git rev-parse HEAD && git status --porcelain=v1", "source": "controller"})
        self.advice_store.append_event("repository_identity", {"actor_call": 0, "result": identity})
        self._snapshot("initial")
        try:
            return super().run(task, **kwargs)
        finally:
            self._snapshot("final")
            self.advice_store.append_event("run_finished", {"actor_call": self.n_calls,
                "patch_id": self.patch_id, "duration_seconds": time.monotonic() - self.started,
                "stop_reason": self.messages[-1].get("extra", {}).get("exit_status") if self.messages else "error"})
            self.finished = True
            self.save(self.config.output_path)

    def query(self):
        capacity = self.config.step_limit == 0 or self.n_calls < self.config.step_limit
        time_remaining = self.config.wall_time_limit_seconds == 0 or time.time() - self._start_time < self.config.wall_time_limit_seconds
        if capacity and time_remaining and self.policy.scheduled(self.n_calls, self.config.step_limit, self.last_checkpoint):
            self.last_checkpoint = self.n_calls
            guidance = self._consult()
            self.add_messages(self.model.format_message(role="user", content=guidance,
                              extra={"advice_id": self.pending_advice} if self.pending_advice else {}))
        if self.policy.phase3 and capacity and time_remaining:
            self._inspection_checkpoint()
            if not self.interventions.inspection_checkpoint:
                self._recovery_checkpoint()
        # DefaultAgent owns call counting and billing, including format-error calls.
        self.model.actor_call = self.n_calls + 1
        self.model.pending_advice = self.pending_advice
        self.model.advice_records = self.advice_records
        self.model.observation_ids = self.observation_ids
        self.model.patch_ids = self.visible_patch_ids
        self.model.advice_enabled = self.policy.actor_requested
        self.model.phase3_enabled = self.policy.phase3
        self.model.recovery_enabled = self.policy.reviews_recovery
        if self.policy.actor_requested:
            self.add_messages(self.model.format_message(role="user", content=self._capacity_status()))
        return super().query()

    def _capacity_status(self):
        remaining = max(0, self.config.step_limit - self.n_calls) if self.config.step_limit else "unlimited"
        return (f"Actor calls remaining: {remaining}\nAdvisor consultations: {self.advisor_calls} used / "
                f"{self.policy.max_calls} maximum; {max(0, self.policy.max_calls - self.advisor_calls)} remaining")

    def _deny_advice(self, reason):
        self.advice_denials[reason] = self.advice_denials.get(reason, 0) + 1
        self.advice_store.append_event("advice_denied", {"actor_call": self.n_calls, "reason": reason})
        return f"Advice unavailable ({reason}); continue independently."

    def execute_actions(self, message):
        actions = message.get("extra", {}).get("actions", [])
        try:
            if len(actions) != 1:
                raise ValueError("Exactly one native action is required")
            action = actions[0]
            validate_action(action["name"], {key: value for key, value in action.items() if key not in {"name", "tool_call_id"}},
                            advice_enabled=self.policy.actor_requested, phase3_enabled=self.policy.phase3, recovery_enabled=self.policy.reviews_recovery)
            decision = action.get("advice_decision")
            validate_advice_context(action, pending_advice=self.pending_advice,
                                   records=self.advice_records, observation_ids=self.observation_ids, patch_ids=self.visible_patch_ids)
            if self.interventions:
                if self.policy.reviews_recovery:
                    self.interventions.validate_recovery(action)
                if self.interventions.inspection_checkpoint:
                    if action["name"] not in {"propose_plan", "report_checkpoint"}:
                        raise ValueError("Respond to the inspection checkpoint with propose_plan or report_checkpoint before continuing")
                    if action["name"] == "report_checkpoint" and "inspection" not in action:
                        raise ValueError("Inspection checkpoint requires inspection explanation, approach, alternatives, and missing_evidence")
                if action["name"] == "propose_plan":
                    self.interventions.validate_plan(action)
        except (ValueError, KeyError, TypeError) as error:
            self.advice_store.append_event("actor_schema_error", {"actor_call": self.n_calls, "error": str(error)})
            # A parsed assistant tool call is already in the transcript; close it
            # with a tool error rather than leaving an unmatched call in context.
            corrections = self.model.format_observation_messages(message,
                [{"output": f"Schema error: {error}", "returncode": -1, "exception_info": "schema_validation"}])
            raise FormatError(*corrections) from error
        self.state.update(action["state"])
        if self.interventions:
            fresh = self.interventions.action(action, self.n_calls)
            if action.get("progress"):
                self.advice_store.append_event("progress_report", {"actor_call": self.n_calls,
                    "report": action["progress"], "new_observable_evidence": fresh,
                    "semantic_assessment": "actor_reported"})
        self.advice_store.append_event("actor_action", {"actor_call": self.n_calls,
            "action": action, "state": self.state, "patch_id": self.patch_id})
        if decision:
            self.advice_store.append_event("advice_decision", {"actor_call": self.n_calls, **decision})
            record = self.advice_records[decision["advice_id"]]
            record.update(decision=decision, status="rejected" if decision["decision"] == "reject" else "acknowledged")
            self._advice_status(record)
            self.pending_advice = None
        for progress in action.get("advice_progress", []):
            record = self.advice_records[progress["advice_id"]]
            record.update(status=progress["status"], progress=progress)
            self._advice_status(record)
        if self.policy.phase3 and action["name"] in {"propose_plan", "report_checkpoint"}:
            return self._phase3_action(message, action)
        if action["name"] == "request_advice":
            self.advice_requests += 1
            self.advice_store.append_event("actor_advice_request", {"actor_call": self.n_calls,
                "question": action["question"], "evidence_ids": action["evidence_ids"]})
            if self.policy.phase3:
                reasons = ["advice_requested"]
                if self.interventions.checkpoint and self.policy.reviews_recovery:
                    reasons += self.interventions.checkpoint["reasons"]
                guidance = self._phase3_consult(action["question"], reasons, action["evidence_ids"], automatic=False)
                self._finish_checkpoint()
            elif self.config.step_limit and self.n_calls >= self.config.step_limit:
                guidance = self._deny_advice("no_actor_capacity")
            elif self.config.wall_time_limit_seconds and time.time() - self._start_time >= self.config.wall_time_limit_seconds:
                guidance = self._deny_advice("no_time_capacity")
            else:
                guidance = self._consult(question=action["question"], reason="advice_requested",
                    evidence_ids=action["evidence_ids"])
            observations = self.model.format_observation_messages(message,
                [{"output": guidance, "returncode": 0, "exception_info": ""}],
                self.get_template_vars())
            return self.add_messages(*observations)
        if action["name"] == "submit":
            self.advice_store.append_event("submission", {"actor_call": self.n_calls,
                "summary": action["summary"], "validation_refs": action["validation_refs"], "patch_id": self.patch_id})
            raise Submitted({"role": "exit", "content": action["summary"],
                             "extra": {"exit_status": "Submitted", "submission": action["summary"]}})
        observation_id = f"observation-{self.n_calls}"
        launch_hash = self.patch_hash
        output = self.env.execute({"command": action["command"], "observation_id": observation_id,
                                   "patch_id": self.patch_id})
        # The environment records the complete combined stdout/stderr and return code.
        self.advice_store.append_event("raw_observation", {"actor_call": self.n_calls,
            "observation_id": observation_id, "result": output})
        visible = dict(output)
        text = str(output.get("output", ""))
        if len(text) > self.max_observation_chars:
            half = (self.max_observation_chars - 80) // 2
            visible["output"] = text[:half] + f"\n[... {len(text) - 2 * half} chars omitted; full output retained ...]\n" + text[-half:]
        self._snapshot("after_action")
        self.observation_ids.add(observation_id)
        for record in self.advice_records.values():
            if record["status"] in OPEN_ADVICE_STATUSES:
                record["actions"].append({"actor_call": self.n_calls, "command": action["command"],
                    "observation_id": observation_id, "patch_id": self.patch_id,
                    "returncode": output.get("returncode"), "exception_info": output.get("exception_info", "")})
                record["status"] = "in_progress"
                self._advice_status(record)
        visible["extra"] = {"observation_id": observation_id, "patch_id": self.patch_id,
                            "output_truncated": len(text) > self.max_observation_chars}
        visible = self.advice_store.redact(visible)
        self.visible_patch_ids.add(self.patch_id)
        self.visible_evidence[observation_id] = (
            f"returncode={visible.get('returncode')}; exception_info={visible.get('exception_info', '')}; "
            f"output_truncated={len(text) > self.max_observation_chars}\n{visible.get('output', '')}")
        if self.interventions:
            classified = action
            if launch_hash != self.patch_hash and action.get("check"):
                classified = {**action, "check": {**action["check"], "kind": "unknown"}}
            record = self.interventions.observation(classified, visible, self.n_calls, observation_id,
                                                    launch_hash, visible.get("output", ""))
            if record:
                self.advice_store.append_event("check_result", {**record,
                    "completion_source_hash": self.patch_hash, "semantic_assessment": "actor_reported"})
        observations = self.model.format_observation_messages(message, [visible], self.get_template_vars())
        # Advertise the evidence ID without changing the saved raw command output.
        for observation in observations:
            observation["content"] = f"[{observation_id}; {self.patch_id}]\n" + observation["content"]
            open_ids = [key for key, record in self.advice_records.items() if record["status"] in OPEN_ADVICE_STATUSES]
            if open_ids:
                observation["content"] += f"\nOpen advice: {json.dumps(open_ids)}. Include advice_progress for each ID in your next action; cite recorded observations to resolve it."
        self.advice_store.append_event("visible_observation", {"actor_call": self.n_calls,
            "observation_id": observation_id, "patch_id": self.patch_id, "messages": observations})
        return self.add_messages(*observations)

    def _phase3_awaiting_decision(self):
        return self.pending_advice is not None

    def _phase3_consult(self, question, reasons, evidence_ids, *, automatic=True):
        reasons = list(dict.fromkeys(reasons))
        refs = list(dict.fromkeys(evidence_ids))[:8]
        related = [key for key, record in self.advice_records.items()
                   if record["status"] in OPEN_ADVICE_STATUSES]
        observed = {entry["observation_id"] for key in related
                    for entry in self.advice_records[key]["actions"]}
        new_refs = [ref for ref in refs if ref in observed]
        denied = None
        if self._phase3_awaiting_decision():
            denied = "pending_advice"
        elif self.config.step_limit and self.config.step_limit - self.n_calls < self.policy.min_actor_calls_remaining:
            denied = "no_actor_capacity"
        elif self._command_time_remaining() is not None and self._command_time_remaining() <= 0:
            denied = "no_time_capacity"
        self.advice_store.append_event("intervention", {"actor_call": self.n_calls,
            "reasons": reasons, "question": question, "evidence_ids": refs,
            "automatic": automatic, "plan_id": self.interventions.plan and self.interventions.plan["plan_id"],
            "patch_id": self.patch_id, "denied": denied,
            "related_advice_ids": related, "new_observation_ids": new_refs})
        if denied:
            return self._deny_advice(denied)
        before = self.advisor_calls
        try:
            guidance = self._consult(question=question, reason="+".join(reasons), evidence_ids=refs,
                                     related_advice_ids=related, new_observation_ids=new_refs)
        except AdvisorUnavailable:
            # Phase 3 advice is optional. Retain failed usage and allow independent
            # continuation; Phase 1/2 preserve their original failure behavior.
            self.advice_store.append_event("advisor_unavailable_continuation", {
                "actor_call": self.n_calls, "reasons": reasons})
            guidance = "Advisor unavailable; the failed consultation was recorded. Continue independently."
        self.advice_store.append_event("intervention_result", {"actor_call": self.n_calls,
            "reasons": reasons, "advice_id": f"advice-{self.advisor_calls}" if self.advisor_calls > before else None,
            "delivered": self.pending_advice is not None, "guidance": guidance,
            "related_advice_ids": related})
        if self.policy.reviews_recovery:
            self.recovery_delivered = self.pending_advice is not None
        return guidance

    def _finish_checkpoint(self):
        collector = self.interventions
        if self.policy.reviews_recovery:
            collector.finish_checkpoint(self.n_calls, reviewed=self.recovery_delivered,
                                         attempted=True,
                                         disposition='reviewed' if self.recovery_delivered else 'review_unavailable')
            self.recovery_delivered = False
            return
        if collector.checkpoint:
            collector.handled_recovery.add(collector.checkpoint["key"])
        collector.checkpoint = None
        collector.checkpoint_announced = False
        collector.fallback_after = None
        collector.fallback_ready = False
        collector.last_progress_call = self.n_calls
        collector.high_count = collector.conflict_count = 0
        collector.failure_count = 0

    def _phase3_action(self, message, action):
        collector = self.interventions
        reasons = []
        if action["name"] == "propose_plan":
            previous = collector.plan
            inspection_pending = collector.inspection_checkpoint is not None
            plan = collector.publish_plan(action)
            self.state["hypothesis"] = plan["hypothesis"]
            self.advice_store.append_event("plan_proposed", {"actor_call": self.n_calls, **plan})
            question = ("Is this causal hypothesis reasonable given the evidence? Critique the proposed approach, "
                        "identify assumptions and missing behavior, and recommend a falsifying check.\n"
                        + json.dumps({"current_plan": plan, "previous_plan": previous}))
            if self.policy.reviews_plans and plan["key"] not in collector.reviewed_plans:
                reasons.append("plan_review")
                if inspection_pending:
                    reasons.append("inspection_checkpoint")
            if collector.checkpoint and self.policy.reviews_recovery:
                reasons += collector.checkpoint["reasons"]
            if reasons:
                guidance = self._phase3_consult(question, reasons, action["evidence_ids"])
                if "plan_review" in reasons:
                    collector.reviewed_plans.add(plan["key"])
                self._finish_checkpoint()
            else:
                guidance = f"Recorded {plan['plan_id']}; automatic review disabled or this plan already reviewed. Continue with your next check."
            guidance = f"Current plan: {plan['plan_id']}. Reference this ID in previous_plan_id when revising.\n" + guidance
        elif collector.inspection_checkpoint:
            collector.record_inspection(action, self.n_calls)
            self.advice_store.append_event("inspection_response", collector.inspection_responses[-1])
            if self.policy.reviews_plans:
                question = ("Review this tentative investigation direction. Insufficient evidence is valid; "
                            "assess competing explanations and recommend the most useful discriminating next check.\n"
                            + json.dumps({key: action[key] for key in ("inspection", "question", "next_check")}))
                guidance = self._phase3_consult(question, ["inspection_checkpoint"], action["evidence_ids"])
            else:
                guidance = "Inspection reasoning recorded locally; automatic plan review is disabled. Execute the proposed next check."
            self._finish_checkpoint()
        else:
            self.advice_store.append_event("checkpoint_response", {"actor_call": self.n_calls, "action": action})
            if not collector.checkpoint:
                guidance = "No recovery checkpoint is pending. Continue, or request_advice for a focused question."
            elif self.policy.reviews_recovery:
                guidance = self._recovery_response(action)
            elif action["question"].strip() and self.policy.reviews_recovery:
                guidance = self._phase3_consult(action["question"], collector.checkpoint["reasons"], action["evidence_ids"])
                self._finish_checkpoint()
            else:
                collector.checkpoint["next_check"] = action["next_check"]
                collector.checkpoint["evidence_ids"] = action["evidence_ids"]
                collector.fallback_after = self.n_calls
                guidance = "Execute the proposed next check. The controller will assess new evidence before another review."
                if not self.policy.reviews_recovery:
                    guidance = ("No advisor consultation occurred: automatic recovery review is disabled. "
                                "Execute the proposed next check, or use request_advice for an unresolved question.")
                    self._finish_checkpoint()
        observations = self.model.format_observation_messages(message,
            [{"output": guidance, "returncode": 0, "exception_info": ""}], self.get_template_vars())
        return self.add_messages(*observations)

    def _inspection_checkpoint(self):
        collector = self.interventions
        if self._phase3_awaiting_decision() or collector.inspection_checkpoint or not collector.inspection_due(self.n_calls):
            return
        collector.inspection_checkpoint = {"actor_call": self.n_calls, "reason": "inspection_checkpoint"}
        self.advice_store.append_event("inspection_checkpoint", collector.inspection_checkpoint)
        self.add_messages(self.model.format_message(role="user", content=(
            "Inspection checkpoint: publish your tentative hypothesis and approach now using propose_plan if actionable, "
            "or respond with report_checkpoint including inspection.explanation, inspection.approach, "
            "inspection.alternatives, inspection.missing_evidence, question, next_check, and evidence_ids. "
            "Insufficient evidence is valid. Identify supporting/contradicting observations and the next discriminating check. "
            "Respond before further execution or submission; certainty is not required.")))

    def _recovery_checkpoint(self):
        collector = self.interventions
        if self._phase3_awaiting_decision():
            return
        if self.policy.reviews_recovery:
            collector.refresh_checkpoint_support(self.n_calls, self.state, self.patch_hash)
            if collector.checkpoint and not collector.checkpoint['reasons']:
                collector.finish_checkpoint(self.n_calls, disposition='supported_concern_recorded')
                return
        if self.policy.reviews_recovery and collector.fallback_ready and collector.checkpoint_resolved():
            self.advice_store.append_event('recovery_resolution', {'actor_call':self.n_calls,
                'checkpoint':collector.checkpoint, 'assessment':'collector_progress_or_actor_evidence'})
            collector.finish_checkpoint(self.n_calls, disposition='resolved_by_experiment')
            return
        if collector.fallback_ready and collector.checkpoint:
            if self.policy.reviews_recovery and self.state.get('blocker') == 'environment':
                self.advice_store.append_event('recovery_deferred', {'actor_call':self.n_calls,
                    'reason':'environment_blocker', 'checkpoint':collector.checkpoint})
                collector.fallback_ready = False
                return
            if self.policy.reviews_recovery and 'concern_evidence_required' in collector.checkpoint['reasons']:
                self.advice_store.append_event('recovery_deferred', {'actor_call':self.n_calls,
                    'reason':'missing_concern_evidence', 'checkpoint':collector.checkpoint})
                collector.fallback_ready = False
                collector.fallback_after = self.n_calls
                self.add_messages(self.model.format_message(role='user', content=(
                    'Recovery concern still needs a focused question and visible evidence. '
                    'Use report_checkpoint to supply them before automatic concern advice.')))
                return
            refs = list(dict.fromkeys([*collector.checkpoint.get("evidence_ids", []), collector.last_observation]))
            question = ("The next experiment did not establish progress. Review the current hypothesis and "
                        "explain what discriminating check should resolve the blocker. Next experiment: "
                        + collector.checkpoint.get("next_check", "unknown"))
            guidance = (self._phase3_consult(question, collector.checkpoint["reasons"], [ref for ref in refs if ref])
                        if self.policy.reviews_recovery else "Local checkpoint recorded; continue independently.")
            self._finish_checkpoint()
            self.add_messages(self.model.format_message(role="user", content=guidance))
            return
        if collector.checkpoint_announced:
            return
        reasons = collector.reasons(self.n_calls, self.state)
        if not reasons:
            return
        key = collector.recovery_key(reasons, self.state, self.patch_hash)
        if key in collector.handled_recovery:
            return
        collector.checkpoint = {"key": key, "reasons": reasons, "actor_call": self.n_calls}
        if self.policy.reviews_recovery:
            collector.checkpoint['concern_id'] = collector.concern and collector.concern['id']
            collector.checkpoint['failure_check_ids'] = sorted({f['check_id'] for f in collector.failures.values()
                if f['source_hash'] == self.patch_hash and f['count'] >= self.policy.failure_threshold})
            collector.checkpoint['failure_checks'] = sorted({(f['check_id'], f['command']) for f in collector.failures.values()
                if f['source_hash'] == self.patch_hash and f['count'] >= self.policy.failure_threshold})
        collector.checkpoint_announced = True
        # If the actor skips the checkpoint response, its next completed command
        # is the one permitted experiment, not an indefinite exemption from review.
        collector.fallback_after = self.n_calls
        self.advice_store.append_event("recovery_checkpoint", collector.checkpoint)
        self.add_messages(self.model.format_message(role="user", content=(
            f"Recovery checkpoint ({', '.join(reasons)}). Use report_checkpoint to give your hypothesis, "
            "supporting/contradicting evidence, a focused unresolved question, and one next_check. "
            "Use an empty question only if the next experiment can resolve the concern independently.")))
        if self.policy.reviews_recovery and collector.concern:
            self.add_messages(self.model.format_message(role='user', content=(
                f"Active recovery concern: {collector.concern['id']}. Supply question and evidence_ids at this "
                "checkpoint if missing. Reference the concern_id with concern_status='resolved' or 'replace' "
                "and visible evidence when closing or replacing it.")))

    def _recovery_response(self, action):
        collector = self.interventions
        checkpoint = collector.checkpoint
        collector.refresh_checkpoint_support(self.n_calls, self.state, self.patch_hash)
        if not checkpoint['reasons']:
            collector.finish_checkpoint(self.n_calls, disposition='supported_concern_recorded')
            return 'Concern evidence recorded. Continue the next check; persistence thresholds still apply.'
        if collector.checkpoint_resolved():
            collector.finish_checkpoint(self.n_calls, disposition='resolved_by_evidence')
            return 'Recovery checkpoint resolved with recorded evidence. Continue independently.'
        question = recovery_question(action['question'])
        if (question and 'concern_evidence_required' not in checkpoint['reasons']
                and self.state.get('blocker') != 'environment'):
            guidance = self._phase3_consult(question, checkpoint['reasons'], action['evidence_ids'])
            self._finish_checkpoint()
            return guidance
        checkpoint.update(next_check=action['next_check'], evidence_ids=action['evidence_ids'])
        collector.fallback_after = self.n_calls
        return 'Execute the next check. New output alone does not resolve the concern; the controller will assess the result.'

    def _advice_status(self, record):
        self.advice_store.append_event("advice_status", {"actor_call": self.n_calls,
            "advice_id": record["advice_id"], "status": record["status"],
            "decision": record.get("decision"), "progress": record.get("progress"),
            "latest_action": record["actions"][-1] if record["actions"] else None})

    def _snapshot(self, reason):
        result = self.env.execute({"command": PATCH_COMMAND, "source": "controller"})
        text = str(result.get("output", ""))
        if result.get("returncode") != 0:
            self.advice_store.append_event("patch_snapshot_failed", {"actor_call": self.n_calls, "reason": reason, "result": result})
            return
        digest = hashlib.sha256(text.encode()).hexdigest()
        if digest != self.patch_hash or reason == "final":
            self.patch_number += 1
            self.patch_id = f"patch-{self.patch_number}"
            self.patch_text, self.patch_hash = text, digest
            self.patch_evidence[self.patch_id] = self.advice_store.redact(text)
            if self.interventions:
                self.interventions.evidence_facts[self.patch_id] = digest
                if self.policy.reviews_recovery:
                    self.interventions.candidate_changed(digest, self.n_calls)
            artifact = self.advice_store.write_artifact(self.patch_id, {"patch": text, "sha256": digest})
            self.advice_store.append_event("patch_snapshot", {"actor_call": self.n_calls,
                "patch_id": self.patch_id, "reason": reason, "artifact": artifact, "sha256": digest})

    def _advisor_record(self, kind, payload):
        if kind == "model_request":
            self.recorded_advisor_requests += 1
        self.advice_store.append_event(kind, {"model_role": "advisor", "actor_call": self.n_calls, **payload})

    def _account_advisor(self, *, completion=None, error=None):
        attempts = (completion.attempts if completion else error.attempts) or ()
        if attempts:
            usages = [attempt["usage"] for attempt in attempts]
            self.advisor_fallbacks += int(len(attempts) > 1)
        elif completion:
            usages = [{**completion.raw_usage, "cost": completion.cost_usd,
                       "prompt_tokens": completion.prompt_tokens, "completion_tokens": completion.completion_tokens}]
        else:
            usages = [error.response.get("usage") or {}]
        self.advisor_api_calls += len(usages)
        for usage in usages:
            cost = amount(usage.get("cost"))
            if cost is None:
                self.unknown_advisor_costs += 1
            else:
                self.advisor_cost += cost
            self.advisor_prompt_tokens += int(usage.get("prompt_tokens") or 0)
            self.advisor_completion_tokens += int(usage.get("completion_tokens") or 0)

    def _consult(self, *, question=SCHEDULED_QUESTION, reason="fixed_interval", evidence_ids=(),
                 related_advice_ids=(), new_observation_ids=()):
        if self.advisor_calls >= self.policy.max_calls:
            return self._deny_advice("call_limit")
        cited = {ref: self.visible_evidence[ref] if ref in self.visible_evidence else self.patch_evidence[ref]
                 for ref in evidence_ids}
        try:
            packet = advice_packet(task=self.extra_template_vars.get("task", ""), messages=self.messages,
                diff=self.patch_text, question=question, actor_calls=self.n_calls,
                actor_limit=self.config.step_limit, max_chars=self.policy.max_context_chars,
                state=self.state, patch_id=self.patch_id, reason=reason, cited_evidence=cited,
                advice_calls_remaining=self.policy.max_calls - self.advisor_calls - 1,
                followthrough=list(self.advice_records.values()),
                related_advice_ids=related_advice_ids, new_observation_ids=new_observation_ids)
        except ValueError:
            return self._deny_advice("context_limit")
        packet = self.advice_store.redact(packet)
        self.advisor_calls += 1
        advice_id = f"advice-{self.advisor_calls}"
        request_id = f"{self.advice_store.run_id}:{advice_id}"
        self.advice_store.append_event("advice_requested", {"actor_call": self.n_calls,
            "advice_id": advice_id, "request_id": request_id, "reason": reason,
            "evidence_ids": list(evidence_ids), "packet": packet,
            "related_advice_ids": list(related_advice_ids)})
        started = time.monotonic()
        try:
            if self.advisor is None:
                self.advisor = Advisor.from_profile(self.advisor_profile,
                    os.environ.get("OPENROUTER_API_KEY") or os.environ.get("MSWEA_API_KEY", ""),
                    self.policy.max_output_tokens, self._advisor_record)
            completion = self.advisor.consult(packet, request_id)
        except (ModelError, ValueError) as error:
            self.advisor_failures += 1
            response = error.response if isinstance(error, ModelError) else {}
            self._account_advisor(error=error if isinstance(error, ModelError) else ModelError(str(error)))
            self.advice_store.append_event("advice_failed", {"actor_call": self.n_calls,
                "advice_id": advice_id, "request_id": request_id, "error": str(error),
                "outcome_unknown": isinstance(error, RequestOutcomeUnknown), "cost_usd": amount((response.get("usage") or {}).get("cost")),
                "attempts": error.attempts if isinstance(error, ModelError) else (),
                "response": response,
                "duration_seconds": time.monotonic() - started})
            raise AdvisorUnavailable(f"Advisor consultation {advice_id} failed: {error}") from error
        self._account_advisor(completion=completion)
        self.advice_store.append_event("advice_received", {"actor_call": self.n_calls,
            "advice_id": advice_id, "request_id": request_id, "completion": completion,
            "duration_seconds": time.monotonic() - started})
        try:
            guidance = validate_guidance(completion.content)
        except (ValueError, TypeError) as error:
            self.advisor_failures += 1
            self.advice_store.append_event("advice_schema_error", {"actor_call": self.n_calls,
                "advice_id": advice_id, "error": str(error)})
            raise AdvisorUnavailable(f"Advisor consultation {advice_id} returned invalid guidance: {error}") from error
        self.pending_advice = advice_id
        self.advice_records[advice_id] = {"advice_id": advice_id, "guidance": guidance,
            "status": "delivered", "actions": [], "related_advice_ids": list(related_advice_ids)}
        self.advice_store.append_event("advice_delivered", {"actor_call": self.n_calls,
            "advice_id": advice_id, "guidance": guidance, "related_advice_ids": list(related_advice_ids)})
        return f"Advisor guidance ({advice_id}):\n{json.dumps(guidance)}\nInclude advice_decision for {advice_id} in your next action."

    def serialize(self, *extra_dicts):
        data = super().serialize(*extra_dicts)
        actor = actor_usage(self.messages, self.n_calls)
        advisor = {"api_calls": self.advisor_calls, "failures": self.advisor_failures,
            "logical_calls": self.advisor_calls,
            "physical_api_calls": max(self.advisor_api_calls, self.recorded_advisor_requests),
            "fallback_calls": self.advisor_fallbacks,
            "unknown_cost_calls": self.unknown_advisor_costs, "observed_cost_usd": str(self.advisor_cost),
            "cost_usd": str(self.advisor_cost) if not self.unknown_advisor_costs else None,
            "prompt_tokens": self.advisor_prompt_tokens, "completion_tokens": self.advisor_completion_tokens}
        total = str(Decimal(actor["cost_usd"]) + Decimal(advisor["cost_usd"])) if actor["cost_usd"] is not None and advisor["cost_usd"] is not None else None
        metrics = {"run_id": self.advice_store.run_id, "policy": self.policy.policy,
            "every_steps": self.policy.every_steps if self.policy.policy == "fixed_interval" else None,
            "max_advice_calls": self.policy.max_calls, "advice_requests": self.advice_requests,
            "advice_calls_remaining": max(0, self.policy.max_calls - self.advisor_calls),
            "advice_denials": self.advice_denials, "budget_enforced": False,
            "actor": actor, "advisor": advisor, "total_cost_usd": total,
            "stop_reason": data["info"].get("exit_status"), "final_patch_id": self.patch_id,
            "duration_seconds": time.monotonic() - self.started}
        metrics["advice_followthrough"] = self.advice_records
        if self.interventions:
            metrics["interventions"] = self.interventions.summary()
        data["info"]["poc_metrics"] = metrics
        self.advice_store.write_summary(metrics)
        return self.advice_store.redact(data)
