import json
import operator
from typing import Annotated, TypedDict

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from kv_cache_agent.observability.events import redact
from kv_cache_agent.observability.logger import RunSession
from kv_cache_agent.observability.nodes import add_logged_node
from kv_cache_agent.observability.tracing import TracingSettings


def records(session):
    return [
        json.loads(line)
        for line in (session.directory / "events.jsonl").read_text().splitlines()
    ]


def test_parallel_nodes_are_correlated_and_events_are_visible_before_exit(tmp_path):
    class State(TypedDict):
        results: Annotated[list[str], operator.add]

    def worker(state):
        return {"results": [state["task_id"]]}

    graph = StateGraph(State)
    add_logged_node(graph, "dispatch", lambda _: {})
    add_logged_node(graph, "worker", worker)
    graph.add_edge(START, "dispatch")
    graph.add_conditional_edges(
        "dispatch",
        lambda _: [
            Send(
                "worker",
                {"task_id": f"t-{i}", "section_ids": [f"s-{i}"], "round_id": 2},
            )
            for i in range(3)
        ],
    )
    graph.add_edge("worker", END)
    with RunSession(
        tmp_path, tracing=TracingSettings(), console=False, heartbeat_seconds=0
    ) as session:
        result = graph.compile().invoke({"results": []})
        live = records(session)
        starts = [e for e in live if e["event"] == "node_start" and e["task_id"]]
        assert {tuple(e["section_ids"]) for e in starts} == {
            ("s-0",),
            ("s-1",),
            ("s-2",),
        }
        assert {e["task_id"] for e in starts} == {"t-0", "t-1", "t-2"}
        for start in starts:
            assert any(
                e["event"] == "node_end"
                and e["invocation_id"] == start["invocation_id"]
                for e in live
            )
        assert len(result["results"]) == 3
    final = records(session)
    assert [e["sequence"] for e in final] == list(range(1, len(final) + 1))
    assert final[-1]["event"] == "run_end"


def test_returned_failure_is_logged_as_failure_and_exception_is_preserved(tmp_path):
    from kv_cache_agent.observability.nodes import logged_node

    def failure(_):
        return {"market_result": {"status": "failed"}}

    def crash(_):
        raise RuntimeError("Bearer FAKE-TOKEN")

    with (
        pytest.raises(RuntimeError),
        RunSession(tmp_path, tracing=TracingSettings(), console=False) as session,
    ):
        logged_node("market", failure)({}, {})
        logged_node("crash", crash)({}, {})
    events = records(session)
    assert any(e["event"] == "node_end" and e["status"] == "failed" for e in events)
    assert any(e["event"] == "node_error" for e in events)
    raw = (session.directory / "events.jsonl").read_text()
    assert "FAKE-TOKEN" not in raw
    assert (
        json.loads((session.directory / "summary.json").read_text())["status"]
        == "failed"
    )


def test_enabled_tracing_requires_settings_before_work_starts(tmp_path):
    with (
        pytest.raises(ValueError, match="LANGSMITH_API_KEY"),
        RunSession(tmp_path, tracing=TracingSettings(enabled=True, project="test")),
    ):
        pytest.fail("Must not enter a misconfigured run")
    assert list(tmp_path.iterdir()) == []


def test_redaction_masks_credentials_but_keeps_token_metrics():
    value = redact(
        {
            "api_key": "SECRET",
            "message": "sk-proj-TEST and tvly-TEST",
            "total_tokens": 10,
        }
    )
    assert value["api_key"] == "[REDACTED]"
    assert "TEST" not in value["message"]
    assert value["total_tokens"] == 10


def test_trace_flush_failure_keeps_local_events_and_summary(monkeypatch, tmp_path):
    from contextlib import nullcontext
    from types import SimpleNamespace
    from unittest.mock import Mock

    from kv_cache_agent.observability import logger as module

    client = Mock()
    client.flush.side_effect = ConnectionError("offline")
    root = SimpleNamespace(trace_id="trace", end=Mock())
    monkeypatch.setattr(TracingSettings, "make_client", lambda _: client)
    monkeypatch.setattr(module, "tracing_context", lambda **_: nullcontext())
    monkeypatch.setattr(module, "trace", lambda *_, **__: nullcontext(root))
    with RunSession(
        tmp_path, tracing=TracingSettings(enabled=True), console=False
    ) as session:
        session.finish("provisional", missing_items=1)
    stored = json.loads((session.directory / "summary.json").read_text())
    assert stored["status"] == "provisional"
    assert stored["tracing_status"] == "degraded"
    assert any(e["event"] == "trace_error" for e in records(session))
