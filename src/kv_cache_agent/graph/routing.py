from kv_cache_agent.graph.state import GlobalState


def route_after_verification(state: GlobalState) -> str:
    result = state.get("verification_result", {})
    if result.get("status") == "needs_retry":
        return "research"
    return "synthesis"


def route_after_quality_check(state: GlobalState) -> str:
    return "end" if state.get("final_report") else "writer"


def route_supervisor_decision(state):
    action = state["decision"].action
    if action in {"research", "revise"}:
        return "prepare_dispatch"
    return "verify_claims" if action == "verify" else "finish"


def dispatch_workers(state, *, parallel=True):
    from langgraph.types import Send

    pending = [
        key
        for key in state["active_task_ids"]
        if key not in state.get("processed_task_ids", [])
        and key not in state.get("deliveries", {})
    ]
    if not parallel:
        pending = pending[:1]
    if not pending:
        raise ValueError("Dispatch has no pending selected tasks")
    return [
        Send(
            "worker",
            {
                "request": state["active_requests"][key],
                "task": state["active_requests"][key].task.model_dump(mode="json"),
            },
        )
        for key in pending
    ]


def route_after_collection(state):
    if state.get("fatal_errors"):
        return "supervisor_decide"
    outstanding = set(state["active_task_ids"]) - set(state["processed_task_ids"])
    return "dispatch" if outstanding else "review_progress"
