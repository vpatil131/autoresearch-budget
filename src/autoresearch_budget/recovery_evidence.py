"""Conservative execution evidence for automatic recovery, not advice validation."""
from __future__ import annotations

import re
import shlex

RECOVERY_EXECUTION_VERSION = "recovery-execution-v3"

# These identify a check that never reached its behavioral assertions. Inspect
# full captured output: the actor-visible excerpt can omit the setup error.
SETUP_FAILURE = re.compile(
    r"ERR_MODULE_NOT_FOUND|MODULE_NOT_FOUND|ModuleNotFoundError:|"
    r"ImportError: (?:No module named|cannot import name)|"
    r"Cannot find (?:package|module) ['\"]|No module named ['\"]|"
    r"command not found|(?:/bin/)?(?:sh|bash|dash):[^\n]*: not found\b|"
    r"/bin/(?:sh|bash):.*Bad substitution|"
    r"Executable doesn't exist at|Could not find (?:Chrome|Chromium|Firefox)|"
    r"Failed to launch the browser process|No test files found|No tests ran",
    re.IGNORECASE,
)


def direct_check_command(command: str) -> bool:
    """Accept a foreground process, optionally preceded by `cd DIR &&`.

    Shell pipelines, sequences, redirects and nested shells can hide the checked
    process's exit status. They still execute normally, but cannot qualify here.
    This is deliberately conservative, not a general-purpose shell parser.
    """
    if "$" in command or "`" in command or "\\\n" in command:
        return False
    try:
        lexer = shlex.shlex(command.strip(), posix=True, punctuation_chars=";&|()<>\n")
        lexer.whitespace = " \t\r"
        lexer.whitespace_split = True
        tokens = list(lexer)
    except ValueError:
        return False
    if tokens and tokens[0] == "cd":
        prefix = 4 if len(tokens) > 1 and tokens[1] == "--" else 3
        if len(tokens) < prefix or tokens[prefix - 1] != "&&":
            return False
        tokens = tokens[prefix:]
    if not tokens or any(re.fullmatch(r"[;&|()<>\n]+", token) for token in tokens):
        return False
    while tokens and re.match(r"^[A-Za-z_][A-Za-z_0-9]*=", tokens[0]):
        tokens = tokens[1:]
    if not tokens:
        return False
    executable = tokens[0].rsplit("/", 1)[-1]
    return executable not in {
        "sh", "bash", "dash", "zsh", "fish", "eval", "exec", "source", ".",
        "if", "then", "for", "while", "until", "case", "function", "env", "xargs",
        "exit", "return", "!", "{", "}", "command", "builtin", "nohup", "setsid",
        "timeout", "sudo", "time",
    }


def recovery_check_evidence(action: dict, result: dict, *, candidate_changed: bool) -> dict:
    """Assess only recovery eligibility; do not change command execution."""
    check = action.get("check", {})
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", str(result.get("output", "")))
    reliable = direct_check_command(action["command"])
    reason = None
    if check.get("kind") != "semantic":
        reason = "not_declared_behavioral"
    elif result.get("timed_out") or result.get("exception_info") or type(result.get("returncode")) is not int:
        reason = "execution_incomplete"
    elif action["state"].get("blocker") == "environment" or SETUP_FAILURE.search(text):
        reason = "observed_environment_failure"
    elif candidate_changed:
        reason = "candidate_changed_during_check"
    elif not reliable:
        reason = "ambiguous_check_exit_status"
    return {"version": RECOVERY_EXECUTION_VERSION, "eligible": reason is None,
            "reason": reason, "exit_status": result.get("returncode"),
            "exit_status_reliable": reliable}
