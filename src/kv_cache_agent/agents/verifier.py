from typing import Any

from kv_cache_agent.graph.state import GlobalState


def evidence_verification_agent(state: GlobalState) -> dict[str, Any]:
    """Check source quality and claim-evidence alignment."""
    return {
        "verification_result": {
            "status": "ok",
            "summary": "TODO: verify evidence cards and source URLs.",
            "evidence_ids": [],
        }
    }
