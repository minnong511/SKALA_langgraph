from kv_cache_agent.graph.state import GlobalState


def test_global_state_is_importable() -> None:
    state: GlobalState = {
        "user_query": "Compare TurboQuant and CXL-based KV Cache",
        "verified_evidence_cards": [],
    }
    assert "user_query" in state
