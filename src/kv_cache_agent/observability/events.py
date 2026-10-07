"""Small, credential-safe execution events."""

import re
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

_SECRET_KEYS = {
    "api_key",
    "authorization",
    "password",
    "access_token",
    "secret",
    "token",
}
_KEY_PATTERN = re.compile(
    r"\b(?:sk-|tvly-|lsv2_)[A-Za-z0-9_-]+|Bearer\s+\S+", re.IGNORECASE
)


def redact(value: Any) -> Any:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if isinstance(value, dict):
        if {"reference", "acquisition", "content_hash", "content"} <= value.keys():
            value = {**value, "content_characters": len(value["content"])}
            value.pop("content")
        return {
            str(k): "[REDACTED]"
            if str(k).lower() in _SECRET_KEYS or str(k).lower().endswith("api_key")
            else redact(v)
            for k, v in value.items()
        }
    if isinstance(value, (tuple, list)):
        return [redact(v) for v in value]
    if isinstance(value, str):
        return _KEY_PATTERN.sub("[REDACTED]", value)
    if value is None or isinstance(value, (int, float, bool)):
        return value
    return f"<{type(value).__name__}>"


def summary(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if not isinstance(value, dict):
        return {"type": type(value).__name__}
    result: dict[str, Any] = {}
    for key in (
        "status",
        "action",
        "reason",
        "error_type",
        "task_id",
        "round_id",
        "section_id",
    ):
        if key in value:
            result[key] = redact(value[key])
    for key, item in value.items():
        if isinstance(item, BaseModel) and hasattr(item, "status"):
            result[f"{key}_status"] = item.status
        if isinstance(item, (list, tuple, dict)):
            result[f"{key}_count"] = len(item)
        if key.endswith("_result") and isinstance(item, dict):
            result[f"{key}_status"] = item.get("status")
        if key == "deliveries" and isinstance(item, dict) and item:
            statuses = [
                delivery.result.status
                for delivery in item.values()
                if hasattr(delivery, "result")
            ]
            if statuses:
                result["deliveries_status"] = (
                    "failed"
                    if "failed" in statuses
                    else (
                        "insufficient_evidence"
                        if "insufficient_evidence" in statuses
                        else "needs_retry"
                        if "needs_retry" in statuses
                        else "ok"
                    )
                )
    return result


class ExecutionEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    sequence: int
    run_id: str
    invocation_id: str | None = None
    parent_invocation_id: str | None = None
    node_path: str
    event: str
    level: str = "INFO"
    task_id: str | None = None
    agent: str | None = None
    round_id: int | None = None
    section_ids: tuple[str, ...] = ()
    technologies: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()
    status: str | None = None
    message: str = ""
    action: str | None = None
    reason: str | None = None
    duration_ms: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)
    langsmith_trace_id: str | None = None
    langsmith_run_id: str | None = None
