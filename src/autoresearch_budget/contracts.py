"""Phase 1 tool contracts, independent of the model SDK."""
from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

CONTRACT_VERSION = "phase1-v2"
ACTOR_SYSTEM = """You solve software engineering issues using the repository and visible checks.
Own the diagnosis, edits, validation, and submission. Every response must call
exactly one native tool: execute or submit. Use execute for shell commands;
each command runs in a new subshell. Use submit to finish; it ends the attempt.
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


def actor_tools(pending_advice: str | None = None, active_advice: tuple[str, ...] = ()) -> list[dict]:
    tools = []
    for name, description, properties, required in (
        ("execute", "Execute a shell command in the repository.", {"command": {"type": "string"}}, ["command"]),
        ("submit", "Finish the attempt and submit the current candidate.", {
            "summary": {"type": "string"},
            "validation_refs": {"type": "array", "items": {"type": "string"}},
        }, ["summary", "validation_refs"]),
    ):
        properties = {**properties, "state": deepcopy(STATE_SCHEMA)}
        required = [*required, "state"]
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
        raise ValueError(f"{field} must be {'a nonempty' if nonempty else 'an'} array of observation IDs")


def validate_action(name: str, args: Any) -> dict:
    if name not in {"execute", "submit"}:
        raise ValueError(f"Unknown phase 1 tool: {name}")
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    fields = {"state", "advice_decision", "advice_progress"} | ({"command"} if name == "execute" else {"summary", "validation_refs"})
    if set(args) - fields:
        raise ValueError(f"Unexpected tool arguments: {sorted(set(args) - fields)}")
    state = args.get("state")
    if not isinstance(state, dict) or set(state) - set(STATE_SCHEMA["properties"]):
        raise ValueError("state must contain only phase, uncertainty, blocker, and optional hypothesis")
    for key in STATE_SCHEMA["required"]:
        if state.get(key) not in STATE_SCHEMA["properties"][key]["enum"]:
            raise ValueError(f"Invalid state.{key}")
    if "hypothesis" in state:
        _text(state["hypothesis"], "state.hypothesis", empty=True)
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
    else:
        _text(args.get("summary"), "summary")
        refs = args.get("validation_refs")
        _refs(refs, "validation_refs")
    return {"name": name, **args}


def validate_advice_context(action: dict, *, pending_advice: str | None,
                            records: dict[str, dict], observation_ids: set[str]) -> None:
    """Gate commands/submission using recorded state, not parsed prose."""
    decision = action.get("advice_decision")
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


def parse_actor_response(response: dict) -> dict:
    choices = response.get("choices") or []
    calls = (choices[0].get("message", {}).get("tool_calls") or []) if choices else []
    if len(calls) != 1:
        raise ValueError("Every actor response requires exactly one execute or submit tool call")
    call = calls[0]
    if call.get("type") != "function" or not call.get("id"):
        raise ValueError("A function tool call with an ID is required")
    function = call.get("function") or {}
    action = validate_action(function.get("name"), json.loads(function.get("arguments", "")))
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
