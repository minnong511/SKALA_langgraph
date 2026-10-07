"""Runtime Send fan-out and post-evaluation conditional routing."""

from langgraph.types import Send

from kv_cache_agent.evidence import claim_fingerprint, source_identity
from kv_cache_agent.observability import node_span, record_event
from kv_cache_agent.schemas.tasks import SubTask


def assign_workers(state):
    control, payload = state["control"], state["payload"]
    if control.get("status") == "stopping":
        return "finalize"
    sends = []
    for value in payload.get("tasks", []):
        task = SubTask.model_validate(value)
        if control["task_status"].get(task.task_id) != "pending":
            continue
        with node_span(
            control["trace_id"],
            "dynamic_fan_out",
            task_id=task.task_id,
            perspective=task.perspective,
        ):
            record_event(
                control["trace_id"],
                "dynamic_fan_out",
                "send",
                task.objective,
                task_id=task.task_id,
                perspective=task.perspective,
                retry_count=task.retry_count,
            )
        sends.append(
            Send(
                "research_worker",
                {
                    "task": task.model_dump(),
                    "trace_id": control["trace_id"],
                    "user_query": payload["user_query"],
                    "target_domain": payload["target_domain"],
                    "limits": control["limits"],
                    "known_source_ids": sorted(
                        {source_identity(c) for c in payload.get("evidence_cards", [])}
                    ),
                    "known_evidence_keys": [
                        claim_fingerprint(c)
                        for c in payload.get("usable_evidence_cards", [])
                    ],
                },
            )
        )
    return sends or "reduce_results"


def route_after_quality_check(state):
    if state["control"].get("status") == "stopping":
        return "stop"
    return state["control"]["decision"]


def route_next(state):
    return "stop" if state["control"].get("status") == "stopping" else "next"
