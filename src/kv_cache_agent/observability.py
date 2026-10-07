"""Small JSON events outside State, correlated with optional LangSmith spans."""

import json
import logging
from contextlib import contextmanager
from datetime import UTC, datetime

from langsmith import get_current_run_tree, trace

logger = logging.getLogger("kv_cache_agent.events")


def record_event(trace_id, node, decision, decision_reason="", **fields):
    event = {
        "trace_id": trace_id,
        "node": node,
        "task_id": fields.pop("task_id", None),
        "perspective": fields.pop("perspective", None),
        "decision": decision,
        "decision_reason": str(decision_reason)[:3000],
        "retry_count": fields.pop("retry_count", 0),
        "timestamp": datetime.now(UTC).isoformat(),
        **fields,
    }
    logger.info(json.dumps(event, ensure_ascii=False))
    run = get_current_run_tree()
    if run is not None:
        run.add_metadata(event)
        # Metadata is searchable but overwrites previous task decisions. Events
        # preserve every Send and route within the same parent run.
        run.add_event({"name": decision, "time": event["timestamp"], "kwargs": event})
    return event


@contextmanager
def node_span(trace_id, node, **metadata):
    # Complete input State is deliberately omitted from custom spans.
    with trace(node, inputs={}, metadata={"trace_id": trace_id, **metadata}):
        yield


def flush_traces():
    """Wait for queued uploads on CLI exit; telemetry never changes graph status."""
    from langsmith.run_trees import get_cached_client
    from langsmith.utils import tracing_is_enabled

    if tracing_is_enabled():
        try:
            get_cached_client().flush(timeout=10)
        except Exception as error:  # noqa: BLE001
            logger.warning("LangSmith flush failed: %s", type(error).__name__)
