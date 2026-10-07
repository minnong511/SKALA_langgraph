"""Shared node registration; nested graphs retain their original behavior."""

import inspect
import traceback
from contextvars import copy_context
from threading import Event, Thread
from time import monotonic
from typing import Any

from langchain_core.runnables import RunnableConfig
from langsmith.run_helpers import get_current_run_tree, tracing_context

from kv_cache_agent.observability.events import summary
from kv_cache_agent.observability.logger import CURRENT_CONTEXT, child_context, emit


def logged_node(name: str, function):
    if getattr(function, "__skala_logged__", False):
        return function

    def wrapped(state: Any, config: RunnableConfig):
        context = child_context(name, state)
        if context is None:
            return (
                function(state, config=config)
                if "config" in inspect.signature(function).parameters
                else function(state)
            )
        token = CURRENT_CONTEXT.set(context)
        started = monotonic()
        stopped = Event()

        def heartbeat():
            while not stopped.wait(context.session.heartbeat_seconds):
                emit(
                    "node_heartbeat",
                    message="호출 결과 대기 중",
                    duration_ms=round((monotonic() - started) * 1000, 2),
                )

        emit("node_start", details=summary(state))
        thread = None
        if context.session.heartbeat_seconds > 0:
            copied = copy_context()
            thread = Thread(target=lambda: copied.run(heartbeat), daemon=True)
            thread.start()
        try:
            metadata = {
                "application_run_id": context.session.run_id,
                "node_invocation_id": context.invocation_id,
                "node_path": context.node_path,
                "task_id": context.task_id,
                "agent": context.agent,
                "round_id": context.round_id,
                "section_ids": list(context.section_ids),
                "technologies": list(context.technologies),
                "criteria": list(context.criteria),
            }
            current_run = get_current_run_tree()
            if current_run is not None:
                current_run.add_metadata(metadata)
            with tracing_context(metadata=metadata):
                result = (
                    function(state, config=config)
                    if "config" in inspect.signature(function).parameters
                    else function(state)
                )
            details = summary(result)
            statuses = [
                value
                for key, value in details.items()
                if key == "status" or key.endswith("_status")
            ]
            failed = "failed" in statuses or (
                isinstance(result, dict) and bool(result.get("error_type"))
            )
            status = (
                "failed"
                if failed
                else "insufficient_evidence"
                if "insufficient_evidence" in statuses
                else "needs_retry"
                if "needs_retry" in statuses or details.get("errors_count", 0)
                else "ok"
            )
            emit(
                "node_end",
                level="ERROR" if failed else "WARNING" if status != "ok" else "INFO",
                status=status,
                duration_ms=round((monotonic() - started) * 1000, 2),
                details=details,
            )
            return result
        except BaseException as error:
            emit(
                "node_cancelled"
                if isinstance(error, (KeyboardInterrupt, SystemExit))
                else "node_error",
                level="ERROR",
                status="failed",
                message=type(error).__name__,
                duration_ms=round((monotonic() - started) * 1000, 2),
                details={
                    "error": str(error),
                    "traceback": "".join(traceback.format_exception(error)),
                },
            )
            raise
        finally:
            stopped.set()
            if thread is not None:
                thread.join(timeout=1)
            CURRENT_CONTEXT.reset(token)

    wrapped.__name__ = function.__name__
    wrapped.__doc__ = function.__doc__
    wrapped.__skala_logged__ = True
    return wrapped


def add_logged_node(graph, name: str, function, *, path: str | None = None):
    return graph.add_node(
        name, logged_node(path or f"{function.__module__}.{name}", function)
    )
