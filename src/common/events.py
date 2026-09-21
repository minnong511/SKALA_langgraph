"""Structured UTC events, serialized across concurrent agent workers."""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import redact

_EMIT_LOCK = threading.Lock()


def _terminal_details(event: str, details: dict[str, Any]) -> str:
    """Display operational facts without echoing arbitrary source/model payloads."""
    fields = []
    names = (
        "elapsed_seconds",
        "duration_ms",
        "status",
        "target_agent",
        "additional_retry",
        "retry_limit",
        "current",
        "total",
        "round",
        "evidence_count",
        "reason",
        "error",
        "errors",
        "path",
    )
    if event == "run_end":
        names += ("unresolved", "paths")
    for name in names:
        if name not in details:
            continue
        value = json.dumps(details[name], ensure_ascii=False, separators=(",", ":"))
        if len(value) > 800:
            value = value[:797] + "..."
        fields.append(f"{name}={value}")
    return " | " + " ".join(fields) if fields else ""


class EventLogger:
    """Append one complete JSON object per line and print the same redacted event."""

    def __init__(
        self, run_dir: Path, run_id: str, *, secrets: list[str] | tuple[str, ...] = (), quiet: bool = False
    ):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.secrets = tuple(secrets)
        self.quiet = quiet
        self.path = self.run_dir / "events.jsonl"

    def emit(
        self,
        event: str,
        message: str,
        *,
        task_id: str = "",
        agent: str = "system",
        attempt: int = 1,
        **details: Any,
    ) -> dict[str, Any]:
        # Timestamp generation belongs inside the lock so file order is chronological.
        with _EMIT_LOCK:
            entry = redact(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                    "run_id": self.run_id,
                    "task_id": task_id,
                    "agent": agent,
                    "attempt": attempt,
                    "event": event,
                    "message": message,
                    "details": details,
                },
                self.secrets,
            )
            encoded = (json.dumps(entry, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600
            )
            with os.fdopen(descriptor, "ab") as handle:
                handle.write(encoded)
                handle.flush()
            if not self.quiet:
                label = f"{entry['agent']}:{entry['task_id']}" if task_id else entry["agent"]
                rendered = f"{entry['timestamp']} [{label}] {entry['event']} (attempt={entry['attempt']}) {entry['message']}"
                rendered = rendered.replace("\r", "\\r").replace("\n", "\\n")
                print(rendered + _terminal_details(event, entry["details"]), flush=True)
            return entry
