from typing import get_type_hints

from kv_cache_agent.graph.state import GlobalState, merge_payload


def test_state_separates_control_and_payload():
    state: GlobalState = {
        "payload": {"user_query": "Compare TurboQuant and CXL-based KV Cache"},
        "control": {"trace_id": "test"},
    }
    assert state["payload"]["user_query"]
    assert "technical_result" not in get_type_hints(GlobalState)


def test_nested_reducer_is_associative_and_idempotent():
    a = {"worker_results": [{"task_id": "a"}], "evidence_cards": [{"evidence_id": "a"}]}
    b = {"worker_results": [{"task_id": "b"}], "evidence_cards": [{"evidence_id": "b"}]}
    c = {"worker_results": [{"task_id": "c"}], "evidence_cards": [{"evidence_id": "c"}]}
    assert merge_payload(merge_payload(a, b), c) == merge_payload(
        a, merge_payload(b, c)
    )
    assert merge_payload(a, a) == a
    assert len(merge_payload(a, b)["evidence_cards"]) == 2
