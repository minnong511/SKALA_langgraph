"""Flush JSONL at each event; log context is local to an invocation."""

import json
import logging
import sys
from contextlib import ExitStack
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4
from zoneinfo import ZoneInfo

from langsmith import trace
from langsmith.run_helpers import get_current_run_tree, tracing_context

from kv_cache_agent.observability.events import ExecutionEvent, redact
from kv_cache_agent.observability.tracing import TracingSettings


@dataclass(frozen=True)
class ExecutionContext:
    session: Any
    node_path: str = "workflow"
    invocation_id: str | None = None
    parent_invocation_id: str | None = None
    task_id: str | None = None
    round_id: int | None = None
    section_ids: tuple[str, ...] = ()
    technologies: tuple[str, ...] = ()
    criteria: tuple[str, ...] = ()
    langsmith_run_id: str | None = None


CURRENT_CONTEXT: ContextVar[ExecutionContext | None] = ContextVar(
    "skala_execution", default=None
)


def emit(event: str, *, level: str = "INFO", message: str = "", **fields):
    context = CURRENT_CONTEXT.get()
    if context is not None:
        context.session.emit(context, event, level=level, message=message, **fields)


class RunSession:
    """One invocation, one file pair, one optional parent LangSmith trace."""

    def __init__(
        self,
        directory: Path,
        *,
        run_id: str | None = None,
        tracing: TracingSettings | None = None,
        console: bool = True,
        heartbeat_seconds: float = 30.0,
        metadata: dict | None = None,
        log_level: str = "INFO",
    ):
        self.run_id = run_id or str(uuid4())
        self.directory = Path(directory) / self.run_id
        self.tracing = tracing or TracingSettings.from_env()
        self.console = console
        self.heartbeat_seconds = heartbeat_seconds
        self.metadata = metadata or {}
        self._logger = logging.getLogger(f"kv_cache_agent.run.{self.run_id}")
        self._logger.propagate = False
        self._logger.setLevel(log_level)
        self._console_handler = logging.StreamHandler(sys.stderr)
        self._console_handler.setFormatter(logging.Formatter("%(message)s"))
        self._lock = RLock()
        self._sequence = 0
        self._stack = ExitStack()
        self.trace_id: str | None = None
        self.trace_url: str | None = None
        self.tracing_status = "disabled"
        self.outcome = "completed"
        self.details: dict = {}

    def __enter__(self):
        # Check before an agent/tool is invoked.
        self.client = self.tracing.make_client()
        self.directory.mkdir(parents=True, exist_ok=True)
        self._file = (self.directory / "events.jsonl").open("a", encoding="utf-8")
        if self.console:
            self._logger.addHandler(self._console_handler)
        self._stack.enter_context(
            tracing_context(
                enabled=self.tracing.enabled,
                client=self.client,
                project_name=self.tracing.project or None,
                metadata={"application_run_id": self.run_id, **redact(self.metadata)},
            )
        )
        if self.client is not None:
            self.root_trace = self._stack.enter_context(
                trace(
                    "skala.workflow",
                    run_type="chain",
                    client=self.client,
                    project_name=self.tracing.project,
                    metadata={
                        "application_run_id": self.run_id,
                        **redact(self.metadata),
                    },
                    inputs=redact(self.metadata),
                )
            )
            self.trace_id = str(self.root_trace.trace_id)
            self.tracing_status = "enabled"
        self._token = CURRENT_CONTEXT.set(
            ExecutionContext(
                session=self,
                task_id=self.metadata.get("task_id"),
                round_id=self.metadata.get("round_id"),
                section_ids=tuple(self.metadata.get("section_ids", ())),
            )
        )
        emit(
            "run_start",
            details={"tracing": self.tracing_status, "project": self.tracing.project},
        )
        return self

    def finish(self, status: str, **details):
        self.outcome = status
        self.details = redact(details)

    def emit(self, context: ExecutionContext, event: str, **fields):
        with self._lock:
            self._sequence += 1
            current_trace = get_current_run_tree()
            record = ExecutionEvent(
                sequence=self._sequence,
                run_id=self.run_id,
                node_path=context.node_path,
                invocation_id=context.invocation_id,
                parent_invocation_id=context.parent_invocation_id,
                task_id=context.task_id,
                round_id=context.round_id,
                section_ids=context.section_ids,
                technologies=context.technologies,
                criteria=context.criteria,
                event=event,
                langsmith_trace_id=self.trace_id,
                langsmith_run_id=str(current_trace.id)
                if current_trace
                else context.langsmith_run_id,
                **redact(fields),
            )
            self._file.write(record.model_dump_json() + "\n")
            self._file.flush()
            if self.console:
                clock = record.timestamp.astimezone(ZoneInfo("Asia/Seoul")).strftime(
                    "%H:%M:%S"
                )
                task = f" task={record.task_id}" if record.task_id else ""
                sections = (
                    f" 절={','.join(record.section_ids)}" if record.section_ids else ""
                )
                scope = " ".join(
                    part
                    for part in (
                        f"회차={record.round_id}"
                        if record.round_id is not None
                        else "",
                        f"기술={','.join(record.technologies)}"
                        if record.technologies
                        else "",
                        f"항목={','.join(record.criteria)}" if record.criteria else "",
                        f"소요={record.duration_ms / 1000:.2f}s"
                        if record.duration_ms is not None
                        else "",
                    )
                    if part
                )
                detail_text = " ".join(
                    f"{key}={str(value)[:180]}"
                    for key, value in record.details.items()
                    if key
                    in {
                        "query",
                        "result_count",
                        "success",
                        "failed",
                        "credits",
                        "reason",
                        "action",
                        "verified",
                        "partial",
                        "unsupported",
                        "errors_count",
                        "pages",
                    }
                )
                self._logger.log(
                    getattr(logging, record.level),
                    f"[{clock} {record.level}] {record.node_path} {event}{task}{sections} "
                    f"{scope} {record.message} {record.status or ''} {detail_text}",
                )

    def __exit__(self, error_type, error, traceback):
        if error is not None:
            self.outcome = "failed"
            self.details["error_type"] = error_type.__name__
        emit(
            "run_end",
            level="ERROR" if self.outcome == "failed" else "INFO",
            status=self.outcome,
            details=self.details,
        )
        CURRENT_CONTEXT.reset(self._token)
        try:
            if self.client is not None:
                self.root_trace.end(outputs={"status": self.outcome, **self.details})
            self._stack.__exit__(error_type, error, traceback)
            if self.client is not None:
                self.client.flush(timeout=5)
                self.trace_url = self.client.get_run_url(run=self.root_trace)
                # A URL is a reference, not proof that ingestion succeeded.
        except Exception as trace_error:  # noqa: BLE001 - preserve local outcome on remote failure
            self.tracing_status = "degraded"
            self.emit(
                ExecutionContext(session=self),
                "trace_error",
                level="WARNING",
                message=type(trace_error).__name__,
            )
        result = {
            "run_id": self.run_id,
            "status": self.outcome,
            "event_count": self._sequence,
            "tracing_status": self.tracing_status,
            "langsmith_trace_id": self.trace_id,
            "trace_url": self.trace_url,
            **self.details,
        }
        (self.directory / "summary.json").write_text(
            json.dumps(redact(result), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self._file.close()
        self._logger.removeHandler(self._console_handler)
        self._console_handler.close()
        return False


def child_context(node_path: str, state: Any) -> ExecutionContext | None:
    parent = CURRENT_CONTEXT.get()
    if parent is None:
        return None
    if hasattr(state, "model_dump"):
        state = state.model_dump(mode="json")
    task = state.get("task", state) if isinstance(state, dict) else {}
    if hasattr(task, "model_dump"):
        task = task.model_dump(mode="json")
    if not isinstance(task, dict):
        task = {}
    claims = state.get("claims", []) if isinstance(state, dict) else []
    claim_sections = tuple(
        dict.fromkeys(c.section_id for c in claims if hasattr(c, "section_id"))
    )
    claim_technologies = tuple(
        dict.fromkeys(c.technology for c in claims if hasattr(c, "technology"))
    )
    claim_criteria = tuple(
        dict.fromkeys(c.criterion for c in claims if hasattr(c, "criterion"))
    )
    traced_run = get_current_run_tree()
    return replace(
        parent,
        node_path=node_path,
        invocation_id=str(uuid4()),
        parent_invocation_id=parent.invocation_id,
        task_id=task.get("task_id", parent.task_id),
        round_id=task.get("round_id", parent.round_id),
        section_ids=tuple(
            task.get("section_ids", claim_sections or parent.section_ids)
        ),
        technologies=tuple(
            task.get("technologies", claim_technologies or parent.technologies)
        ),
        criteria=tuple(task.get("criteria", claim_criteria or parent.criteria)),
        langsmith_run_id=str(traced_run.id) if traced_run else None,
    )
