"""Conditional routing helpers used after Agent 5."""

from state import AgentState


def route_after_validation(state: AgentState) -> str:
    """Route to synthesis, targeted retry, or failure.

    Suggested conditional edges:
    - ``synthesis`` -> Agent 6
    - ``retry`` -> Supervisor retry dispatcher
    - ``failed`` -> terminal/error handler
    """

    if state.get("validation_passed"):
        return "synthesis"
    if state.get("retry_targets"):
        return "retry"

    validation = state.get("validation_result") or {}
    if validation.get("retry_exhausted"):
        # Continue with verified/uncertain claims and preserve limitations.
        return "synthesis"
    return "failed"
