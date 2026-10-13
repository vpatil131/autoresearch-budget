"""Inspect external Harbor task packages without exposing verifier contents."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tomllib

from .trial_pool import select_tasks
from .provenance import package_hash


def task_identity(directory: Path) -> dict:
    directory = directory.resolve()
    for name in ("task.toml", "instruction.md"):
        if not (directory / name).is_file():
            raise ValueError(f"task package is incomplete: {directory.name}; missing {name}")
    task = tomllib.loads((directory / "task.toml").read_text())
    if not isinstance(task.get("agent"), dict) or not isinstance(task.get("environment"), dict):
        raise ValueError(f"task.toml needs [agent] and [environment]: {directory.name}")
    if not (directory / "environment").is_dir() or not (directory / "tests").is_dir():
        raise ValueError(f"task package needs environment/ and tests/: {directory.name}")
    digest = package_hash(directory)
    count = sum(p.is_file() for p in directory.rglob("*"))
    return {"task_id": directory.name, "package_sha256": digest, "file_count": count}


def inspect_tasks(task_root: Path, manifest: Path) -> dict:
    selected = select_tasks(None, manifest)
    raw = json.loads(manifest.read_text())
    return {"schema_version": 1, "task_root": str(task_root.expanduser().resolve()),
            "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "source": raw.get("source"),
            "tasks": [task_identity(task_root.expanduser() / name) for name in selected]}
