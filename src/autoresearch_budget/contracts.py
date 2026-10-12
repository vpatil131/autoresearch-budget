"""Actor/advisor tool contracts, independent of the model SDK."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

CONTRACT_VERSION = "phase1-v2"
PHASE2_CONTRACT_VERSION = "phase2-v1"
PHASE3_CONTRACT_VERSION = "phase3-v3"
RECOVERY_CONTRACT_VERSION = "phase3-recovery-v1"
ADVICE_PROTOCOL_VERSION = "advice-messaging-v1"
ACTOR_SYSTEM = """You solve software engineering issues using the repository and visible checks.
Own the diagnosis, edits, validation, and submission. Every response must call
exactly one native tool: execute or submit. Use execute for shell commands;
each command runs in a new subshell. Use submit to finish; it ends the attempt.
Run builds and tests in the foreground. The controller waits for completion and
returns immediately with output and the exit code, up to the command or task
deadline. Avoid background launches, nohup, and sleep-based polling. After a
timeout, inspect the captured output and diagnose the cause before restarting.
Report phase, uncertainty, and blocker in structured state on every action.
Update hypothesis when it changes. Optional advisor guidance arrives every 10
actor calls. Treat it as a proposal. The next action after delivered guidance must
include advice_decision with its advice_id, adopt/modify/reject, and a reason.
For adopt/modify, specify next_action and validation_plan, and immediately execute
a concrete command advancing that plan. For reject, give a concrete reason and
evidence_refs to existing observations. Acknowledgment does not complete advice.
While accepted advice is open, include advice_progress for EACH open advice ID
on every action. For in_progress, reason explains how this command advances its
plan; evidence_refs can be empty. For validated/blocked/rejected, give a reason
and evidence_refs from commands actually executed under that advice. validated
requires successful command evidence; it is your assessment, not hidden grading.
Do not submit with advice still in_progress, or adopt advice and submit without
acting. Resolve each open recommendation with evidence before submission.
Validate changes using public checks. Reference observation IDs in submit's
validation_refs. Do not use hidden grading results or benchmark solutions.
"""
ACTOR_INSTANCE = """Please solve this issue: {{task}}

Inspect relevant code, reproduce the issue, implement a fix, and validate it.
Use the execute tool for commands and submit for completion. There is no shell
submission command. Report concise reasoning and structured state with each tool.
"""


def actor_prompts(policy: str, every_steps: int, max_calls: int) -> tuple[str, str]:
    system = ACTOR_SYSTEM.replace("every 10", f"every {every_steps}")
    instance = ACTOR_INSTANCE
    if policy == "actor_only":
        system = system.replace(f"Optional advisor guidance arrives every {every_steps}\nactor calls.",
                                "Advisor consultation is disabled for this attempt.")
    elif policy == "advice_requested":
        system = ACTOR_SYSTEM.replace("exactly one native tool: execute or submit.",
                                     "exactly one native tool: execute, request_advice, or submit.")
        system = system.replace("Optional advisor guidance arrives every 10\nactor calls.", f"""You decide whether and when to consult the advisor. Advice is optional.
Consider requesting advice early, after initial inspection, when competing
hypotheses remain or the expected API contract is unclear. Before a substantial
edit or expensive validation run, consult if an unresolved question could change
your implementation or validation plan. If two checks fail without clarifying
the cause, consider consulting with those results instead of repeating the same
approach. You may consult at other points when a focused uncertainty makes
progress difficult. Do not wait until most actor calls are spent to raise an
uncertainty you already recognize. Request only when you can name a question
whose answer could change your next action; there is no minimum number of calls.
Ask a specific question and cite observation or patch IDs you have seen in
evidence_ids.
Do not delegate the whole task. You own the solution. You have at most {max_calls}
logical consultations for this attempt; the controller reports remaining
capacity each turn. Each request consumes one actor turn. There are no scheduled
consultations. If the limit is reached, continue independently. Acknowledge
pending guidance before requesting more advice.""")
        system = system.replace("how this command advances its", "how this action advances its")
        instance = instance.replace("Use the execute tool for commands and submit for completion.",
            "Use execute for commands, request_advice for focused consultation, and submit for completion.")
    if policy in {"phase3_control", "plan_review", "recovery_review", "hybrid"}:
        system, instance = actor_prompts("advice_requested", every_steps, max_calls)
        system = system.replace("execute, request_advice, or submit.",
                                "execute, request_advice, propose_plan, report_checkpoint, or submit.")
        system += """
Phase 3: Before implementing your first actionable diagnosis, use propose_plan
with hypothesis, approach, evidence_ids, falsifying_check, and validation_plan.
Every plan includes previous_plan_id (empty for the initial plan, current plan ID
for a revision) and change_reason explaining the initial choice or what changed
and why the cited evidence changes your solution. Routine environment or test
fixture fixes are progress, not new solution hypotheses.
Publish materially revised causal explanations or approaches with propose_plan;
retain the same wording for an unchanged plan. Do not count rewording as progress.
The controller records every plan and may return advice before implementation.
Do not wait for certainty to expose an actionable tentative approach. When the
controller sends an inspection checkpoint, respond with propose_plan if ready,
or report_checkpoint with inspection.explanation, inspection.approach,
inspection.alternatives, inspection.missing_evidence, question, next_check, and
evidence_ids. Insufficient evidence is a valid explanation: state the competing
possibilities and discriminating next check. You must respond before further
execution or submission. This is a reasoning checkpoint, not a diagnosis deadline.
On a local recovery checkpoint, use report_checkpoint with a focused unresolved
question (empty only if one next experiment can resolve it), next_check, and
observed evidence_ids. The controller may consult, or allow that experiment.
On execute, optionally identify a check using check.id, kind (semantic,
environment, unknown), and expected (success or failure). A failing reproduction
on original code is expected failure, not a failed fix. Use environment for
setup/dependency failures; do not label arbitrary shell errors semantic tests.
Report high uncertainty/contradictions with state.uncertainty_question and
state.evidence_refs. Optionally report progress with kind, summary, and evidence_refs
from completed observations or visible patches. Kinds: localization,
discriminating_check, candidate, validation, environment_recovery. New IDs,
reworded plans, and repeated searches do not themselves establish progress.
Acknowledge delivered guidance before another consultation. Once acknowledged,
you may request follow-up advice or publish a revised plan while earlier advice
is in progress. Cite new observations when available and explain what remains
unresolved. Include advice_progress for every open advice ID; a follow-up does
not validate earlier guidance. Resolve every open item with command evidence
before submission. Advice is optional; after a denied/unavailable review continue
independently.
"""
        modes = {"phase3_control": "Automatic plan and recovery advice are disabled.",
                 "plan_review": "Automatic plan review is enabled; recovery advice is disabled.",
                 "recovery_review": "Automatic recovery review is enabled; plan advice is disabled.",
                 "hybrid": "Automatic plan and recovery review are enabled."}
        system += modes[policy]
        if policy in {'recovery_review', 'hybrid'}:
            system += """
Recovery-v1: The controller tracks activity separately from behavioral progress.
Automatic recovery has at most one consultation by default. While advice is open,
another automatic recovery review requires a completed behavioral experiment,
not more setup or inspection. Label actual experiments with check.kind='semantic'
and a stable check.id (including expected failing baseline controls). Explicit
request_advice remains available for concrete obstacles or unresolved results.
Recovery execution evidence requires a completed direct foreground check, optionally
`cd DIR && CHECK`, run separately from candidate edits. Pipes, redirects, command
sequences, and nested shell wrappers have ambiguous check exit status and cannot
unlock automatic recovery. Keep stdout/stderr capture with the controller. Setup
or module-resolution failures are environment evidence even if labeled semantic.
Commands still execute normally; use a direct rerun only when it is meaningful.
Follow-ups in the same recovery episode supersede earlier recovery IDs only
after adoption; unfinished validation requirements carry forward, not as proven.
A candidate edit permits one completed experiment; edit churn is not validation.
At a recovery checkpoint, supply a concrete question and visible evidence to
support a concern, or use question='' to defer for the stated next experiment.
The controller assigns a stable concern_id. Rephrase its question freely while
the concern is open. Optional state.concern_id references that active concern;
state.concern_status is open, resolved, or replace. Resolve/replace only with
state.evidence_refs or checkpoint evidence_ids explaining the resolution or
replacement. Low uncertainty or successful unrelated checks alone do not close
a concern. Use replace when a different question needs investigation, and publish
a revised plan when the causal explanation or approach materially changes.
Classify regression/validation checks with check.id, kind and expected when
available; inspection commands can remain unclassified. Keep check IDs stable
across reruns. The controller ignores known timestamps/durations in failure
output, while retaining assertion values. Advice delivery does not solve the
concern; execute its next experiment and report the observed outcome.
"""
    return system, instance

STATE_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "phase": {"type": "string", "enum": ["inspect", "diagnose", "implement", "validate"]},
        "uncertainty": {"type": "string", "enum": ["low", "medium", "high", "unknown"]},
        "blocker": {"type": "string", "enum": ["none", "missing_information", "conflicting_evidence", "environment", "unknown"]},
        "hypothesis": {"type": "string"},
    }, "required": ["phase", "uncertainty", "blocker"],
}
DECISION_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "advice_id": {"type": "string"},
        "decision": {"type": "string", "enum": ["adopt", "modify", "reject"]},
        "reason": {"type": "string"},
        "next_action": {"type": "string", "minLength": 1},
        "validation_plan": {"type": "string", "minLength": 1},
        "evidence_refs": {"type": "array", "items": {"type": "string"}, "minItems": 1},
    }, "required": ["advice_id", "decision", "reason"],
    "anyOf": [
        {"properties": {"decision": {"enum": ["adopt", "modify"]}}, "required": ["next_action", "validation_plan"]},
        {"properties": {"decision": {"enum": ["reject"]}}, "required": ["evidence_refs"]},
    ],
}
PROGRESS_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "advice_id": {"type": "string"},
        "status": {"type": "string", "enum": ["in_progress", "validated", "blocked", "rejected"]},
        "reason": {"type": "string", "minLength": 1},
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
    }, "required": ["advice_id", "status", "reason", "evidence_refs"],
}
OPEN_ADVICE_STATUSES = {"acknowledged", "in_progress"}


EVIDENCE_SCHEMA = {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                   "items": {"type": "string", "minLength": 1, "maxLength": 80}}
LOCAL_PROGRESS_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "kind": {"type": "string", "enum": ["localization", "discriminating_check", "candidate", "validation", "environment_recovery"]},
    "summary": {"type": "string", "minLength": 1, "maxLength": 2000},
    "evidence_refs": deepcopy(EVIDENCE_SCHEMA)}, "required": ["kind", "summary", "evidence_refs"]}
CHECK_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    "id": {"type": "string", "minLength": 1, "maxLength": 80},
    "kind": {"type": "string", "enum": ["semantic", "environment", "unknown"]},
    "expected": {"type": "string", "enum": ["success", "failure"]}}, "required": ["id", "kind", "expected"]}
INSPECTION_SCHEMA = {"type": "object", "additionalProperties": False, "properties": {
    key: {"type": "string", "minLength": 1, "maxLength": 2000}
    for key in ("explanation", "approach", "alternatives", "missing_evidence")},
    "required": ["explanation", "approach", "alternatives", "missing_evidence"]}


def bounded_refs(value, field):
    _refs(value, field, nonempty=True)
    if len(value) > 8 or len(set(value)) != len(value) or any(len(ref) > 80 for ref in value):
        raise ValueError(f"{field} requires 1–8 distinct evidence IDs of at most 80 characters")


def validate_phase3_fields(args):
    state = args["state"]
    if "uncertainty_question" in state:
        _text(state["uncertainty_question"], "state.uncertainty_question", empty=True)
        if len(state["uncertainty_question"]) > 4000:
            raise ValueError("state.uncertainty_question exceeds 4000 characters")
    if "evidence_refs" in state:
        bounded_refs(state["evidence_refs"], "state.evidence_refs")
    for key, schema in (("progress", LOCAL_PROGRESS_SCHEMA), ("check", CHECK_SCHEMA), ("inspection", INSPECTION_SCHEMA)):
        if key not in args:
            continue
        value = args[key]
        if not isinstance(value, dict) or set(value) != set(schema["required"]):
            raise ValueError(f"{key} requires {schema['required']}")
        for field, spec in schema["properties"].items():
            if field == "evidence_refs":
                bounded_refs(value[field], f"{key}.{field}")
            else:
                _text(value[field], f"{key}.{field}")
                if "enum" in spec and value[field] not in spec["enum"]:
                    raise ValueError(f"Invalid {key}.{field}")
                if len(value[field]) > spec.get("maxLength", 80):
                    raise ValueError(f"{key}.{field} exceeds its character allowance")


def actor_tools(pending_advice: str | None = None, active_advice: tuple[str, ...] = (), *, advice_enabled=False, phase3_enabled=False, recovery_enabled=False) -> list[dict]:
    tools = []
    definitions = [
        ("execute", "Execute a shell command in the repository.", {"command": {"type": "string"}}, ["command"]),
        ("submit", "Finish the attempt and submit the current candidate.", {
            "summary": {"type": "string"},
            "validation_refs": {"type": "array", "items": {"type": "string"}},
        }, ["summary", "validation_refs"]),
    ]
    if advice_enabled:
        definitions.append(("request_advice", "Ask the advisor a focused question using observed evidence.", {
            "question": {"type": "string", "minLength": 1, "maxLength": 4000},
            "evidence_ids": {"type": "array", "minItems": 1, "maxItems": 8, "uniqueItems": True,
                             "items": {"type": "string", "minLength": 1, "maxLength": 80}},
        }, ["question", "evidence_ids"]))
    if phase3_enabled:
        definitions.extend([
            ("propose_plan", "Publish an actionable hypothesis and approach before implementation.",
             {**{key: {"type": "string", "minLength": 1, "maxLength": 2000}
                  for key in ("hypothesis", "approach", "falsifying_check", "validation_plan")},
              "evidence_ids": deepcopy(EVIDENCE_SCHEMA),
              "previous_plan_id": {"type": "string", "maxLength": 80},
              "change_reason": {"type": "string", "minLength": 1, "maxLength": 2000}},
             ["hypothesis", "approach", "falsifying_check", "validation_plan", "evidence_ids", "previous_plan_id", "change_reason"]),
            ("report_checkpoint", "Answer an inspection or recovery checkpoint with reasoning and a next experiment.",
             {"question": {"type": "string", "maxLength": 4000},
              "next_check": {"type": "string", "minLength": 1, "maxLength": 2000},
              "inspection": deepcopy(INSPECTION_SCHEMA),
              "evidence_ids": deepcopy(EVIDENCE_SCHEMA)}, ["question", "next_check", "evidence_ids"]),
        ])
    for name, description, properties, required in definitions:
        properties = {**properties, "state": deepcopy(STATE_SCHEMA)}
        required = [*required, "state"]
        if phase3_enabled:
            properties["state"]["properties"].update({
                "uncertainty_question": {"type": "string", "maxLength": 4000},
                "evidence_refs": deepcopy(EVIDENCE_SCHEMA)})
            properties["progress"] = deepcopy(LOCAL_PROGRESS_SCHEMA)
            if name == "execute":
                properties["check"] = deepcopy(CHECK_SCHEMA)
            if recovery_enabled:
                properties['state']['properties'].update({
                    'concern_id':{'type':'string','minLength':1,'maxLength':80},
                    'concern_status':{'type':'string','enum':['open','resolved','replace']}})
        if pending_advice:
            schema = deepcopy(DECISION_SCHEMA)
            schema["properties"]["advice_id"]["enum"] = [pending_advice]
            properties["advice_decision"] = schema
            required.append("advice_decision")
        if active_advice:
            schema = deepcopy(PROGRESS_SCHEMA)
            schema["properties"]["advice_id"]["enum"] = list(active_advice)
            properties["advice_progress"] = {"type": "array", "items": schema,
                "minItems": len(active_advice), "maxItems": len(active_advice)}
            required.append("advice_progress")
        tools.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "additionalProperties": False,
                "properties": properties, "required": required},
        }})
    return tools


def _text(value: Any, field: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a {'nonempty ' if not empty else ''}string")


def _refs(value: Any, field: str, *, nonempty: bool = False) -> None:
    if not isinstance(value, list) or any(not isinstance(ref, str) or not ref for ref in value) or nonempty and not value:
        raise ValueError(f"{field} must be {'a nonempty' if nonempty else 'an'} array of evidence IDs")


def validate_action(name: str, args: Any, *, advice_enabled=False, phase3_enabled=False, recovery_enabled=False) -> dict:
    enabled = {"execute", "submit"} | ({"request_advice"} if advice_enabled else set())
    if phase3_enabled:
        enabled |= {"propose_plan", "report_checkpoint"}
    if name not in enabled:
        raise ValueError(f"Unknown or disabled actor tool: {name}")
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    fields = {"state", "advice_decision", "advice_progress"} | {
        "execute": {"command"}, "submit": {"summary", "validation_refs"},
        "request_advice": {"question", "evidence_ids"},
        "propose_plan": {"hypothesis", "approach", "evidence_ids", "falsifying_check", "validation_plan", "previous_plan_id", "change_reason"},
        "report_checkpoint": {"question", "next_check", "evidence_ids", "inspection"}}[name]
    if phase3_enabled:
        fields |= {"progress"}
        if name == "execute":
            fields.add("check")
    if set(args) - fields:
        raise ValueError(f"Unexpected tool arguments: {sorted(set(args) - fields)}")
    state = args.get("state")
    state_fields = set(STATE_SCHEMA["properties"]) | ({"uncertainty_question", "evidence_refs"} if phase3_enabled else set())
    if phase3_enabled and recovery_enabled:
        state_fields |= {'concern_id','concern_status'}
    if not isinstance(state, dict) or set(state) - state_fields:
        raise ValueError("state must contain only phase, uncertainty, blocker, and optional hypothesis")
    for key in STATE_SCHEMA["required"]:
        if state.get(key) not in STATE_SCHEMA["properties"][key]["enum"]:
            raise ValueError(f"Invalid state.{key}")
    if "hypothesis" in state:
        _text(state["hypothesis"], "state.hypothesis", empty=True)
    if phase3_enabled:
        validate_phase3_fields(args)
        if recovery_enabled:
            if 'concern_id' in state:
                _text(state['concern_id'], 'state.concern_id')
                if len(state['concern_id']) > 80:
                    raise ValueError('state.concern_id exceeds its character allowance')
            if 'concern_status' in state and state['concern_status'] not in {'open','resolved','replace'}:
                raise ValueError('Invalid state.concern_status')
    if "advice_decision" in args:
        decision = args["advice_decision"]
        if not isinstance(decision, dict) or set(decision) - set(DECISION_SCHEMA["properties"]) or set(DECISION_SCHEMA["required"]) - set(decision):
            raise ValueError("advice_decision requires advice_id, decision, and reason")
        _text(decision["advice_id"], "advice_id")
        _text(decision["reason"], "reason")
        if decision["decision"] not in {"adopt", "modify", "reject"}:
            raise ValueError("Invalid advice decision")
        if decision["decision"] in {"adopt", "modify"}:
            _text(decision.get("next_action"), "advice_decision.next_action")
            _text(decision.get("validation_plan"), "advice_decision.validation_plan")
            if name != "execute":
                raise ValueError("Adopt/modify requires executing the next action before submission")
        if "evidence_refs" in decision or decision["decision"] == "reject":
            _refs(decision.get("evidence_refs"), "advice_decision.evidence_refs", nonempty=True)
    if "advice_progress" in args:
        progress = args["advice_progress"]
        if not isinstance(progress, list):
            raise ValueError("advice_progress must be an array")
        for item in progress:
            if not isinstance(item, dict) or set(item) != set(PROGRESS_SCHEMA["required"]):
                raise ValueError("advice_progress requires advice_id, status, reason, evidence_refs")
            _text(item["advice_id"], "advice_progress.advice_id")
            _text(item["reason"], "advice_progress.reason")
            if item["status"] not in PROGRESS_SCHEMA["properties"]["status"]["enum"]:
                raise ValueError("Invalid advice_progress.status")
            _refs(item["evidence_refs"], "advice_progress.evidence_refs", nonempty=item["status"] != "in_progress")
    if name == "execute":
        _text(args.get("command"), "command")
    elif name in {"propose_plan", "report_checkpoint"}:
        bounded_refs(args.get("evidence_ids"), "evidence_ids")
        if name == "propose_plan":
            _text(args.get("previous_plan_id"), "previous_plan_id", empty=True)
            _text(args.get("change_reason"), "change_reason")
            if len(args["previous_plan_id"]) > 80 or len(args["change_reason"]) > 2000:
                raise ValueError("Plan predecessor/reason exceeds its character allowance")
        keys = ("hypothesis", "approach", "falsifying_check", "validation_plan") if name == "propose_plan" else ("question", "next_check")
        for key in keys:
            _text(args.get(key), key, empty=key == "question")
            if len(args[key]) > (4000 if key == "question" else 2000):
                raise ValueError(f"{key} exceeds its character allowance")
    elif name == "request_advice":
        _text(args.get("question"), "question")
        _refs(args.get("evidence_ids"), "evidence_ids", nonempty=True)
        ids = args["evidence_ids"]
        if len(args["question"]) > 4000 or len(ids) > 8 or len(set(ids)) != len(ids) or any(len(ref) > 80 for ref in ids):
            raise ValueError("Advice requests require a question of at most 4000 characters and 1–8 distinct evidence IDs of at most 80 characters")
    else:
        _text(args.get("summary"), "summary")
        refs = args.get("validation_refs")
        _refs(refs, "validation_refs")
    return {"name": name, **args}


def advice_turn_requirements(pending_advice: str | None, records: dict[str, dict]) -> str:
    """Describe the advice fields required against the pre-action controller state."""
    active = sorted(key for key, record in records.items() if record["status"] in OPEN_ADVICE_STATUSES)
    if not pending_advice and not active:
        return ""
    lines = ["Advice fields required for this turn (before accepting your action):"]
    if pending_advice:
        lines += [f"Pending acknowledgment ID: {json.dumps(pending_advice)}. Include advice_decision for this ID.",
                  f"Do not include {json.dumps(pending_advice)} in advice_progress this turn; its adopt/modify/reject decision is the acknowledgment."]
    else:
        lines.append("No pending acknowledgment. Omit advice_decision.")
    lines.append(f"Progress IDs required this turn: {json.dumps(active)}.")
    if active:
        lines.append("Include advice_progress with exactly one entry for each listed ID, and no other IDs. Use in_progress with evidence_refs=[] if command evidence is unavailable; blocked and resolved statuses require recorded command observations.")
    else:
        lines.append("Omit advice_progress this turn.")
    if pending_advice:
        lines.append("If adopted/modified, the new ID joins the progress list on the following turn after successful acknowledgment; rejected advice does not join it.")
    return "\n".join(lines)


def actor_schema_correction(error: str, *, pending_advice: str | None, records: dict[str, dict]) -> str:
    """Give both validation paths the same correction without changing controller state."""
    text = (f"Schema error: {error}\nThe rejected action was not executed or accepted; "
            "its advice decision and progress were not recorded. Correct the structured fields "
            "and resend the action.")
    requirements = advice_turn_requirements(pending_advice, records)
    return text + ("\n" + requirements if requirements else "")


def validate_advice_context(action: dict, *, pending_advice: str | None,
                            records: dict[str, dict], observation_ids: set[str], patch_ids: set[str] | None = None) -> None:
    """Gate commands/submission using recorded state, not parsed prose."""
    decision = action.get("advice_decision")
    if action["name"] in {"request_advice", "propose_plan", "report_checkpoint"}:
        if pending_advice:
            raise ValueError("Acknowledge pending guidance before requesting more advice")
        if set(action["evidence_ids"]) - (observation_ids | (patch_ids or set())):
            raise ValueError("Advice request contains unknown or unseen evidence IDs")
    for refs in (action["state"].get("evidence_refs", []), action.get("progress", {}).get("evidence_refs", [])):
        if set(refs) - (observation_ids | (patch_ids or set())):
            raise ValueError("Progress or uncertainty contains unknown or unseen evidence IDs")
    if pending_advice:
        if not decision or decision["advice_id"] != pending_advice:
            raise ValueError(f"advice_decision for {pending_advice} is required")
    elif decision:
        raise ValueError("No advisor guidance is awaiting a decision")
    if decision and set(decision.get("evidence_refs", [])) - observation_ids:
        raise ValueError("Advice decision contains unknown observation IDs")
    active = {key: value for key, value in records.items() if value["status"] in OPEN_ADVICE_STATUSES}
    progress = action.get("advice_progress", [])
    ids = [item["advice_id"] for item in progress]
    if len(ids) != len(set(ids)) or set(ids) != set(active) or (not active and "advice_progress" in action):
        raise ValueError(f"advice_progress must cover exactly the open advice IDs: {sorted(active)}")
    for item in progress:
        refs = set(item["evidence_refs"])
        if refs - observation_ids:
            raise ValueError("Advice progress contains unknown observation IDs")
        evidence = {entry["observation_id"]: entry for entry in active[item["advice_id"]]["actions"]}
        if refs - evidence.keys():
            raise ValueError("Advice progress evidence must come from commands executed under that advice")
        if item["status"] == "validated" and any(evidence[ref]["returncode"] != 0 or evidence[ref].get("exception_info") for ref in refs):
            raise ValueError("Validated advice requires successful command evidence")
        if action["name"] == "submit" and item["status"] == "in_progress":
            raise ValueError("Resolve open advice with evidence before submission")
    if action["name"] == "submit" and set(action["validation_refs"]) - observation_ids:
        raise ValueError("validation_refs contains unknown observation IDs")


def parse_actor_response(response: dict, *, advice_enabled=False, phase3_enabled=False, recovery_enabled=False) -> dict:
    choices = response.get("choices") or []
    calls = (choices[0].get("message", {}).get("tool_calls") or []) if choices else []
    if len(calls) != 1:
        raise ValueError("Every actor response requires exactly one enabled native tool call")
    call = calls[0]
    if call.get("type") != "function" or not call.get("id"):
        raise ValueError("A function tool call with an ID is required")
    function = call.get("function") or {}
    action = validate_action(function.get("name"), json.loads(function.get("arguments", "")), advice_enabled=advice_enabled, phase3_enabled=phase3_enabled, recovery_enabled=recovery_enabled)
    return {**action, "tool_call_id": call["id"]}


def validate_guidance(content: str) -> dict:
    value = json.loads(content)
    fields = {"assessment", "gap", "next_action", "validation", "assumptions"}
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError("Advisor response requires assessment, gap, next_action, validation, assumptions")
    for field in fields - {"assumptions"}:
        _text(value[field], field)
    if not isinstance(value["assumptions"], list):
        raise ValueError("assumptions must be an array")
    for assumption in value["assumptions"]:
        _text(assumption, "assumption")
    return value
