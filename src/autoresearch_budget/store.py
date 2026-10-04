"""Append-only run artifacts stored outside task containers."""

from __future__ import annotations

import json
import os
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
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def write_metadata(self, metadata: dict[str, Any]) -> None:
        self._write_json_atomic(self.run_dir / "metadata.json", metadata)

    def append_event(self, kind: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        encoded = json.dumps(record, default=_json_default, sort_keys=True) + "\n"
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())

    def write_summary(self, summary: dict[str, Any]) -> None:
        self._write_json_atomic(self.run_dir / "summary.json", summary)

    @staticmethod
    def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)

