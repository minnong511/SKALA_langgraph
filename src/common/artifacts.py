"""Immutable, private, per-run artifacts and shared credential redaction."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from typing import Any

REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "refresh_token",
    "id_token",
    "token",
    "authorization",
    "proxy_authorization",
    "password",
    "passwd",
    "secret",
    "client_secret",
    "credentials",
    "credential",
    "private_key",
    "secret_key",
    "cookie",
    "set_cookie",
    "openai_api_key",
    "tavily_api_key",
}


def _plain(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    return value


def redact(value: Any, secrets: Sequence[str] = ()) -> Any:
    """Return a JSON-compatible copy, concealing credentials at every depth."""
    needles = sorted({str(secret) for secret in secrets if secret}, key=len, reverse=True)

    def visit(item: Any) -> Any:
        item = _plain(item)
        if isinstance(item, Mapping):
            result = {}
            for key, nested in item.items():
                key = str(key)
                normalized = re.sub(r"[-\s]", "_", key).lower()
                sensitive = normalized in _SENSITIVE_KEYS or normalized.endswith(
                    ("_api_key", "_password", "_secret", "_access_token", "_private_key")
                )
                result[visit(key)] = REDACTED if sensitive else visit(nested)
            return result
        if isinstance(item, (list, tuple, set, frozenset)):
            return [visit(nested) for nested in item]
        if isinstance(item, str):
            for secret in needles:
                item = item.replace(secret, REDACTED)
            return item
        if item is None or isinstance(item, (int, float, bool)):
            return item
        return visit(str(item))

    return visit(value)


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{label} must be a nonempty safe file identifier")
    return value


def atomic_write_exclusive(destination: Path, content: bytes) -> Path:
    """Publish a fully written file atomically; an existing path is never replaced."""
    descriptor, temporary_name = tempfile.mkstemp(prefix=".pending-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


class ArtifactStore:
    """Store every task attempt separately and refuse reuse of an existing run."""

    def __init__(self, output_dir: str | Path, run_id: str, secrets: Sequence[str] = ()):
        self.secrets = tuple(secrets)
        output = Path(output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        self.run_dir = output / _identifier(run_id, "run_id")
        self.run_dir.mkdir(mode=0o700, exist_ok=False)

    def _destination(self, name: str | Path) -> Path:
        relative = Path(name)
        if relative.is_absolute() or not relative.parts or any(part == ".." for part in relative.parts):
            raise ValueError("Artifact paths must stay within the run directory")
        destination = self.run_dir / relative
        # Resolve existing symlinks before creating any directories or temporary files.
        if not destination.resolve().is_relative_to(self.run_dir):
            raise ValueError("Artifact paths must stay within the run directory")
        destination.parent.mkdir(parents=True, exist_ok=True)
        return destination

    def save_json(self, name: str | Path, value: Any) -> Path:
        payload = json.dumps(redact(value, self.secrets), ensure_ascii=False, indent=2, allow_nan=False)
        return atomic_write_exclusive(self._destination(name), (payload + "\n").encode("utf-8"))

    def save_text(self, name: str | Path, text: str) -> Path:
        return atomic_write_exclusive(self._destination(name), redact(text, self.secrets).encode("utf-8"))

    def save_result(self, result: Any) -> Path:
        value = _plain(result)
        if not isinstance(value, Mapping):
            raise TypeError("An agent result must expose a mapping or model_dump()")
        task_id = _identifier(value.get("task_id"), "task_id")
        attempt = value.get("attempt")
        if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
            raise ValueError("attempt must be a positive integer")
        return self.save_json(Path("tasks") / task_id / f"attempt-{attempt}.json", value)
