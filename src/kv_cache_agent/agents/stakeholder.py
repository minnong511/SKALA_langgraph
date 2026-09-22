from typing import Any

from kv_cache_agent.graph.state import GlobalState


def stakeholder_evaluation_agent(state: GlobalState) -> dict[str, Any]:
    """Research stakeholder incentives, benefits, and concerns."""
    return {
        "stakeholder_result": {
            "status": "insufficient_evidence",
            "summary": "TODO: analyze cloud, software, hardware, and customer perspectives.",
            "evidence_ids": [],
        },
        "evidence_cards": [],
    }
