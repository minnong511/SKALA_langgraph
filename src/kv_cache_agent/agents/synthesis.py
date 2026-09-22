from typing import Any

from kv_cache_agent.graph.state import GlobalState


def synthesis_agent(state: GlobalState) -> dict[str, Any]:
    """Compare verified results across all perspectives."""
    return {
        "synthesis_result": {
            "status": "insufficient_evidence",
            "summary": "TODO: build comparison matrix and balanced trade-off analysis.",
            "evidence_ids": [],
        }
    }
