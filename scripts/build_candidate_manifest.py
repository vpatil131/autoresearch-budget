#!/usr/bin/env python3
"""Build a safe-to-inspect manifest from a downloaded SWE-rebench Harbor package.

The Harbor package contains oracle and test patches. This script records only task
identity, resource requirements, public problem metadata, and verifier shape. It
never copies patch bodies or solution scripts into the generated manifest.
"""

from __future__ import annotations

import argparse
import json
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


def utc_timestamp(milliseconds: int | str | None) -> str | None:
    if milliseconds is None:
        return None
    if isinstance(milliseconds, str) and not milliseconds.isdigit():
        parsed = datetime.fromisoformat(milliseconds)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc).isoformat()
    return datetime.fromtimestamp(int(milliseconds) / 1000, tz=timezone.utc).isoformat()


def first_from_image(dockerfile: Path) -> str:
    for line in dockerfile.read_text().splitlines():
        if line.startswith("FROM "):
            return line.removeprefix("FROM ").split()[0]
    raise ValueError(f"No FROM instruction in {dockerfile}")


def task_record(task_dir: Path) -> dict[str, object]:
    config = json.loads((task_dir / "tests" / "config.json").read_text())
    with (task_dir / "task.toml").open("rb") as file:
        task_toml = tomllib.load(file)

    environment = task_toml.get("environment", {})
    install_config = config.get("install_config") or {}
    instruction = (task_dir / "instruction.md").read_text()

    return {
        "task_id": config["instance_id"],
        "repo": config["repo"],
        "pull_number": config.get("pull_number"),
        "created_at": utc_timestamp(config.get("created_at")),
        "language": config.get("language"),
        "base_commit": config.get("base_commit"),
        "image": first_from_image(task_dir / "environment" / "Dockerfile"),
        "resource_request": {
            "cpus": environment.get("cpus"),
            "memory_mb": environment.get("memory_mb"),
            "storage_mb": environment.get("storage_mb"),
        },
        "timeouts_sec": {
            "agent": (task_toml.get("agent") or {}).get("timeout_sec"),
            "verifier": (task_toml.get("verifier") or {}).get("timeout_sec"),
            "build": environment.get("build_timeout_sec"),
        },
        "verifier": {
            "test_command": install_config.get("test_cmd"),
            "log_parser": install_config.get("log_parser"),
            "fail_to_pass_count": len(config.get("FAIL_TO_PASS") or []),
            "pass_to_pass_count": len(config.get("PASS_TO_PASS") or []),
        },
        "public_problem": {
            "title": config.get("pr_title"),
            "instruction_chars": len(instruction),
            "modified_files_in_reference_pr": (config.get("meta") or {}).get(
                "num_modified_files"
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        nargs="?",
        default=Path("data/harbor/swe-rebench-07-2026"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifests/july_2026_manifest.json"),
    )
    args = parser.parse_args()

    task_dirs = sorted(path.parent.parent for path in args.dataset.glob("*/tests/config.json"))
    tasks = [task_record(task_dir) for task_dir in task_dirs]
    language_counts = Counter(task["language"] for task in tasks)
    repo_counts = Counter(task["repo"] for task in tasks)

    document = {
        "source": "ibragim-badertdinov/swe-rebench-07-2026@latest",
        "generated_from_local_package": str(args.dataset),
        "oracle_material_included": False,
        "task_count": len(tasks),
        "language_counts": dict(sorted(language_counts.items())),
        "repository_count": len(repo_counts),
        "tasks": tasks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n")
    print(
        f"Wrote {len(tasks)} tasks across {len(repo_counts)} repositories "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()
