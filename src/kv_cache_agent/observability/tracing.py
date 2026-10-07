"""Tracing configuration is explicit; offline tests never construct a client."""

import os
from dataclasses import dataclass
from functools import wraps

from langsmith import Client, traceable

from kv_cache_agent.observability.events import redact, summary


@dataclass(frozen=True)
class TracingSettings:
    enabled: bool = False
    api_key: str = ""
    project: str = ""
    endpoint: str | None = None
    workspace_id: str | None = None

    @classmethod
    def from_env(cls):
        return cls(
            enabled=os.getenv("LANGSMITH_TRACING", "false").lower() in {"true", "1"},
            api_key=os.getenv("LANGSMITH_API_KEY", ""),
            project=os.getenv("LANGSMITH_PROJECT", ""),
            endpoint=os.getenv("LANGSMITH_ENDPOINT") or None,
            workspace_id=os.getenv("LANGSMITH_WORKSPACE_ID") or None,
        )

    def make_client(self) -> Client | None:
        self.validate()
        if not self.enabled:
            return None

        return Client(
            api_key=self.api_key,
            api_url=self.endpoint,
            workspace_id=self.workspace_id,
            hide_inputs=redact,
            hide_outputs=redact,
            hide_metadata=redact,
        )

    def validate(self) -> None:
        if not self.enabled:
            return
        if not self.api_key or not self.project:
            raise ValueError(
                "LANGSMITH_API_KEY and LANGSMITH_PROJECT are required for tracing"
            )


def traced_tool(name: str):
    """Use for plain SDK/tool functions, not already traced graph nodes."""

    def decorate(function):
        traced = traceable(
            name=name,
            run_type="tool",
            process_inputs=redact,
            process_outputs=lambda output: (
                summary(output) if isinstance(output, dict) else redact(output)
            ),
        )(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            # Explicit settings/context controls upload; no implicit client creation.
            return traced(*args, **kwargs)

        return wrapped

    return decorate
