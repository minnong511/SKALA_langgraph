from typing import Any

from kv_cache_agent.graph.state import GlobalState


def cloud_domain_agent(state: GlobalState) -> dict[str, Any]:
    """Evaluate both technologies only in the cloud serving domain."""
    return {
        "cloud_domain_result": {
            "status": "insufficient_evidence",
            "summary": "TODO: evaluate cloud serving, concurrency, SLO, and cost scenarios.",
            "evidence_ids": [],
        },
        "evidence_cards": [],
    }
