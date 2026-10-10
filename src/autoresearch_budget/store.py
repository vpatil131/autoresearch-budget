"""Append-only run artifacts stored outside task containers."""

from __future__ import annotations

import json
import os
import re
import uuid
from threading import RLock
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


class RunStore:
    def __init__(self, run_dir: str | Path, *, run_id: str | None = None,
                 secrets: tuple[str, ...] = ()):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.run_id = run_id or self.run_dir.name
        self.secrets = tuple(value for value in secrets if value)
        self._lock = RLock()
        self.sequence = 0
        if self.events_path.exists():
            with self.events_path.open() as stream:
                self.sequence = sum(1 for _ in stream)

    def redact(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: "[REDACTED]" if re.search(r"authorization|api[_-]?key|access[_-]?token|password", str(key), re.I)
                    else self.redact(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            for secret in self.secrets:
                value = value.replace(secret, "[REDACTED]")
        return value

    def capture_file(self, source: Path, destination: Path) -> None:
        """Preserve post-run files and redact credentials in text records."""
        source, destination = Path(source), Path(destination)
        raw = source.read_bytes()
        try:
            value = self.redact(raw.decode("utf-8"))
            destination.write_text(value, encoding="utf-8")
        except UnicodeDecodeError:
            destination.write_bytes(raw)

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self._write_json_atomic(self.run_dir / "metadata.json", self.redact({"run_id": self.run_id, **metadata}))

    def append_event(self, kind: str, payload: dict[str, Any]) -> str:
        with self._lock:
            return self._append_event(kind, payload)

    def _append_event(self, kind: str, payload: dict[str, Any]) -> str:
        self.sequence += 1
        event_id = f"{self.run_id}:event:{self.sequence}"
        record = self.redact({
            "run_id": self.run_id, "event_id": event_id, "sequence": self.sequence,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        })
        encoded = json.dumps(record, default=_json_default, sort_keys=True) + "\n"
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return event_id

    def write_artifact(self, name: str, value: Any) -> str:
        """Immutable, JSON-encoded evidence; never overwrite an earlier snapshot."""
        directory = self.run_dir / "artifacts"
        directory.mkdir(exist_ok=True)
        path = directory / f"{name}-{uuid.uuid4().hex}.json"
        with path.open("x", encoding="utf-8") as stream:
            json.dump(self.redact(value), stream, default=_json_default, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        return str(path.relative_to(self.run_dir))

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json_atomic(self.run_dir / "summary.json", self.redact(summary))

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
