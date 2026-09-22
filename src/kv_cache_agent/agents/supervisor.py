from typing import Any

from kv_cache_agent.graph.state import GlobalState


def supervisor_agent(state: GlobalState) -> dict[str, Any]:
    """Plan work and route the worker agents."""
    plan = {
        "technologies": ["TurboQuant", "CXL-based"],
        "perspectives": ["technical", "market", "stakeholder", "cloud_domain"],
        "search_questions": {},
    }
    return {
        "research_plan": plan,
        "control": {
            "status": "researching",
            "next_agent": "technical",
            "retry_count": {},
            "errors": [],
        },
    }
