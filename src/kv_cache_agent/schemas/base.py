"""Validated boundaries shared by the foundation and future agents."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, ConfigDict


class Contract(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )


def fingerprint(value: Any) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    encoded = json.dumps(value, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(encoded.encode()).hexdigest()


def stable_id(kind: str, value: Any) -> str:
    return f"{kind}-{fingerprint(value)[:24]}"
