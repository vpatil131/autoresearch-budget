"""Phase 1 tool contracts, independent of the model SDK."""
from __future__ import annotations

import json
from typing import Any

CONTRACT_VERSION = "phase1-v1"
ACTOR_SYSTEM = """You solve software engineering issues using the repository and visible checks.
Own the diagnosis, edits, validation, and submission. Every response must call
exactly one native tool: execute or submit. Use execute for shell commands;
each command runs in a new subshell. Use submit to finish; it ends the attempt.
Report phase, uncertainty, and blocker in structured state on every action.
Update hypothesis when it changes. Optional advisor guidance arrives every 10
actor calls. Treat it as a proposal. The next action after delivered guidance must
include advice_decision with its advice_id, adopt/modify/reject, and a reason.
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
    }, "required": ["advice_id", "decision", "reason"],
}


def actor_tools() -> list[dict]:
    tools = []
    for name, description, properties, required in (
        ("execute", "Execute a shell command in the repository.", {"command": {"type": "string"}}, ["command"]),
        ("submit", "Finish the attempt and submit the current candidate.", {
            "summary": {"type": "string"},
            "validation_refs": {"type": "array", "items": {"type": "string"}},
        }, ["summary", "validation_refs"]),
    ):
        tools.append({"type": "function", "function": {
            "name": name, "description": description,
            "parameters": {"type": "object", "additionalProperties": False,
                "properties": {**properties, "state": STATE_SCHEMA, "advice_decision": DECISION_SCHEMA},
                "required": [*required, "state"]},
        }})
    return tools


def _text(value: Any, field: str, *, empty: bool = False) -> None:
    if not isinstance(value, str) or (not empty and not value.strip()):
        raise ValueError(f"{field} must be a {'nonempty ' if not empty else ''}string")


def validate_action(name: str, args: Any) -> dict:
    if name not in {"execute", "submit"}:
        raise ValueError(f"Unknown phase 1 tool: {name}")
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    fields = {"state", "advice_decision"} | ({"command"} if name == "execute" else {"summary", "validation_refs"})
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
        if not isinstance(decision, dict) or set(decision) != set(DECISION_SCHEMA["required"]):
            raise ValueError("advice_decision requires advice_id, decision, and reason")
        _text(decision["advice_id"], "advice_id")
        _text(decision["reason"], "reason")
        if decision["decision"] not in {"adopt", "modify", "reject"}:
            raise ValueError("Invalid advice decision")
    if name == "execute":
        _text(args.get("command"), "command")
    else:
        _text(args.get("summary"), "summary")
        refs = args.get("validation_refs")
        if not isinstance(refs, list) or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("validation_refs must be an array of observation IDs")
    return {"name": name, **args}


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
