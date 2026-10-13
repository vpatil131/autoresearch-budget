"""Offline demonstration of the production advisory controller and evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile


def _response(index, name, args):
    return {"id": f"scripted-{index}", "model": "mock/actor", "choices": [{"message": {
        "role": "assistant", "content": "Scripted offline demonstration.", "tool_calls": [{
            "id": f"tool-{index}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(args)}}]}, "finish_reason": "tool_calls"}],
        "usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}


def run_demo(output: Path) -> dict:
    from .actor_model import StructuredActorModel
    from .advisory_agent import AdvisoryAgent
    from .advisor import Advisor
    from .models import OpenRouterClient
    from .poc_environment import PocEnvironment
    from minisweagent.config import get_config_from_spec

    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError("demo output must be new or empty; existing evidence is preserved")
    output.mkdir(parents=True, exist_ok=True)
    state = {"phase": "inspect", "uncertainty": "medium", "blocker": "none"}
    with tempfile.TemporaryDirectory(prefix="autoresearch-demo-") as temporary:
        repo = Path(temporary)
        def git(*args):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
        git("init", "-q"); git("config", "user.email", "demo@example.test"); git("config", "user.name", "Offline demo")
        (repo / "calc.py").write_text("def add(a, b):\n    return a - b\n")
        git("add", "calc.py"); git("commit", "-qm", "Scripted fixture")
        python = shlex.quote(sys.executable)
        responses = [
            _response(1, "execute", {"command": "cat calc.py", "state": state}),
            _response(2, "propose_plan", {"hypothesis": "Addition uses subtraction", "approach": "Use addition",
                "falsifying_check": "Check add(2, 3) equals 5", "validation_plan": "Run a public assertion",
                "previous_plan_id": "", "change_reason": "The inspected operator explains the issue",
                "evidence_ids": ["observation-1"], "state": state}),
            _response(3, "execute", {"command": python + " -c " + shlex.quote(
                "from pathlib import Path; Path('calc.py').write_text('def add(a, b):\\n    return a + b\\n')") + " && " + python +
                " -c " + shlex.quote("from calc import add; assert add(2, 3) == 5; print('Regression passed')"),
                "state": {**state, "phase": "validate"}, "advice_decision": {
                    "advice_id": "advice-1", "decision": "adopt", "reason": "The observed operator supports the guidance",
                    "next_action": "Change the operator and run the assertion", "validation_plan": "Assert add(2, 3) equals 5"}}),
            _response(4, "submit", {"summary": "Fixed addition and passed a public regression", "validation_refs": ["observation-3"],
                "state": {**state, "phase": "validate"}, "advice_progress": [{"advice_id": "advice-1", "status": "validated",
                    "reason": "The edited implementation passes the linked check", "evidence_refs": ["observation-3"]}]})]
        def actor_transport(request, timeout):
            if not responses:
                raise ValueError("Scripted demo exhausted its responses")
            return 200, json.dumps(responses.pop(0)).encode()
        def advisor_transport(request, timeout):
            guidance = {"assessment": "The operator contradicts addition", "gap": "No public regression yet",
                        "next_action": "Replace subtraction with addition", "validation": "Assert add(2, 3) equals 5",
                        "assumptions": ["The function should perform arithmetic addition"]}
            return 200, json.dumps({"id": "scripted-advice", "model": "mock/advisor", "choices": [{"message": {
                "content": json.dumps(guidance)}, "finish_reason": "stop"}],
                "usage": {"cost": 0, "prompt_tokens": 0, "completion_tokens": 0}}).encode()
        actor = StructuredActorModel(model_name="openrouter/mock/actor", backend=OpenRouterClient(
            api_key="offline-placeholder", model="mock/actor", allow_paid=True, transport=actor_transport),
            observation_template=get_config_from_spec("mini")["model"]["observation_template"])
        advisor = Advisor(OpenRouterClient(api_key="offline-placeholder", model="mock/advisor", allow_paid=True,
                                           transport=advisor_transport), 256)
        agent = AdvisoryAgent(actor, PocEnvironment(cwd=str(repo)), advisor=advisor,
            advice={"policy": "plan_review", "max_calls": 1, "run_id": "offline-demo"},
            cost_limit=0, step_limit=8, output_path=output / "agent/mini-swe-agent.trajectory.json")
        actor.backend.recorder = actor._record
        advisor.backend.recorder = agent._advisor_record
        outcome = agent.run("Fix add(a, b) so it adds its arguments. Validate using a public assertion.")
        if outcome.get("exit_status") != "Submitted" or agent.advice_records.get("advice-1", {}).get("status") != "validated":
            raise ValueError("Demo failed to complete validated advice; inspect its saved evidence")
        (output / "patch.diff").write_text(agent.patch_text)
        metrics = agent.serialize()["info"]["poc_metrics"]
        metadata = {"schema_version": 1, "kind": "scripted-demo", "task_id": "demo-addition", "profile": "scripted",
                    "network": False, "budget_enforced": False, "configuration": {"policy": "plan_review"}}
        (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        (output / "summary.json").write_text(json.dumps({"status": "completed", "return_code": 0,
            "metrics": metrics, "total_cost_usd": "0", "budget_enforced": False}, indent=2) + "\n")
    timeline = ["Actor inspected calc.py", "Actor proposed an evidence-linked plan", "Advisor recommended a fix and check",
                "Actor adopted the guidance, edited the file, and passed the check", "Actor linked validation evidence and submitted"]
    return {"output": str(output), "network": False, "total_cost_usd": "0", "timeline": timeline,
            "stop_reason": "Submitted", "advice_status": "validated"}
