from kv_cache_agent.graph.state import GlobalState


def route_after_verification(state: GlobalState) -> str:
    result = state.get("verification_result", {})
    if result.get("status") == "needs_retry":
        return "research"
    return "synthesis"


def route_after_quality_check(state: GlobalState) -> str:
    return "end" if state.get("final_report") else "writer"
