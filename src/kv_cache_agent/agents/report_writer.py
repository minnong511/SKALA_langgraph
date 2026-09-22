from typing import Any

from kv_cache_agent.graph.state import GlobalState


def report_writer_agent(state: GlobalState) -> dict[str, Any]:
    """Write the final report from verified synthesis results."""
    return {
        "final_report": "TODO: generate the cited multi-perspective evaluation report."
    }
