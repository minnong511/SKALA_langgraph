from typing import Any

from kv_cache_agent.graph.state import GlobalState


def market_evaluation_agent(state: GlobalState) -> dict[str, Any]:
    """Use Tavily to research market and adoption evidence."""
    return {
        "market_result": {
            "status": "insufficient_evidence",
            "summary": "TODO: search market, adoption, cost, and barrier evidence.",
            "evidence_ids": [],
        },
        "evidence_cards": [],
    }
