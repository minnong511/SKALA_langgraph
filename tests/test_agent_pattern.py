"""상위 Supervisor의 동적 분기와 보고서 평가 루프를 API 없이 검증한다."""

from copy import deepcopy

import pytest

from kv_cache_agent.agents import supervisor as supervisor_module
from kv_cache_agent.graph.workflow import build_workflow

WORKERS = ("technical", "market", "stakeholder", "cloud_domain")


@pytest.fixture(autouse=True)
def _no_live_supervisor_call(monkeypatch):
    monkeypatch.setattr(supervisor_module, "OPENAI_API_KEY", "")


def _card(worker: str) -> dict:
    return {
        "evidence_id": f"{worker}-1",
        "perspective": worker,
        "technology": "TurboQuant" if worker != "cloud_domain" else "CXL-based",
        "source_url": f"https://example.org/{worker}",
        "verification_status": "verified",
    }


def _nodes(*, market_fails_first: bool = False, quality_fails_first: bool = False):
    calls: list[str] = []
    market_queries: list[list[str]] = []
    writer_feedback: list[list[str]] = []

    def worker(name):
        def run(state):
            calls.append(name)
            if name == "market":
                market_queries.append(
                    state.get("research_plan", {}).get("search_questions", {}).get("market", [])
                )
            failed = name == "market" and market_fails_first and calls.count("market") == 1
            return {
                f"{name}_result": {
                    "agent_name": name, "status": "insufficient_evidence" if failed else "ok",
                    "summary": name, "evidence_ids": [] if failed else [f"{name}-1"],
                    "limitations": [], "errors": [], "payload": {},
                },
                "evidence_cards": [] if failed else [_card(name)],
            }
        return run

    def verifier(state):
        calls.append("verifier")
        cards = deepcopy(state.get("evidence_cards", []))
        return {
            "verification_result": {"agent_name": "verifier", "status": "ok", "payload": {}},
            "verified_evidence_cards": cards,
            "usable_evidence_cards": cards,
        }

    def synthesis(state):
        calls.append("synthesis")
        return {"synthesis_result": {"agent_name": "synthesis", "status": "ok", "payload": {}}}

    def writer(state):
        calls.append("report_writer")
        writer_feedback.append(state.get("quality_feedback", []))
        return {"final_report": "# SUMMARY\n본문 [1]\n# REFERENCE\n[1] 출처"}

    def quality(state):
        calls.append("quality_eval")
        fail = quality_fails_first and calls.count("quality_eval") == 1
        return {
            "quality_result": {
                "passed": not fail, "groundedness": not fail,
                "neutrality": True, "bias_control": True,
                "perspective_coverage": True,
                "feedback": ["시장 출처를 본문에 더 연결하세요."] if fail else [],
                "status": "revise" if fail else "passed",
            },
            "quality_feedback": ["시장 출처를 본문에 더 연결하세요."] if fail else [],
        }

    nodes = {name: worker(name) for name in WORKERS}
    nodes.update(
        {"verifier": verifier, "synthesis": synthesis,
         "report_writer": writer, "quality_eval": quality}
    )
    return nodes, calls, market_queries, writer_feedback


def test_supervisor_waits_for_verified_four_perspectives():
    nodes, calls, _, _ = _nodes()
    result = build_workflow(node_overrides=nodes).invoke({"user_query": "KV cache 평가"})
    assert all(calls.count(worker) == 1 for worker in WORKERS)
    assert calls.index("verifier") < calls.index("synthesis") < calls.index("report_writer")
    assert result["control"]["evidence_ready"] is True
    assert result["quality_result"]["passed"] is True
    assert result["trace_id"]


def test_insufficient_market_evidence_is_sent_back_only_to_market():
    nodes, calls, market_queries, _ = _nodes(market_fails_first=True)
    result = build_workflow(node_overrides=nodes).invoke({"user_query": "KV cache 평가"})
    assert calls.count("market") == 2
    assert all(calls.count(worker) == 1 for worker in WORKERS if worker != "market")
    assert calls.count("verifier") == 2
    assert market_queries[1] != market_queries[0]
    # 재작업 사유를 그대로 검색하지 않고 기술 관련 질의를 사용해야 한다.
    assert "완료되지 않았음" not in " ".join(market_queries[1])
    assert "TurboQuant" in market_queries[1][0]
    assert result["control"]["retry_count"]["market"] == 2


def test_quality_failure_rewrites_report_once_with_feedback():
    nodes, calls, _, feedback = _nodes(quality_fails_first=True)
    result = build_workflow(node_overrides=nodes).invoke({"user_query": "KV cache 평가"})
    assert calls.count("report_writer") == 2
    assert calls.count("quality_eval") == 2
    assert feedback[0] == []
    assert "시장 출처" in feedback[1][0]
    assert result["control"]["report_revision"] == 2


def test_research_retry_limit_terminates_with_explicit_uncertainty():
    calls: list[str] = []

    def failed_worker(name):
        def run(state):
            calls.append(name)
            return {
                f"{name}_result": {"agent_name": name, "status": "insufficient_evidence"},
                "evidence_cards": [],
            }
        return run

    nodes = {name: failed_worker(name) for name in WORKERS}
    nodes["verifier"] = lambda state: {
        "verification_result": {"status": "insufficient_evidence", "payload": {}},
        "verified_evidence_cards": [], "usable_evidence_cards": [],
    }
    nodes["synthesis"] = lambda state: {
        "synthesis_result": {"status": "insufficient_evidence"}
    }
    nodes["report_writer"] = lambda state: {"final_report": "# SUMMARY\n한계\n# REFERENCE"}
    nodes["quality_eval"] = lambda state: {
        "quality_result": {"status": "needs_review", "passed": False, "feedback": ["근거 부족"]}
    }
    result = build_workflow(node_overrides=nodes).invoke({"user_query": "KV cache 평가"})
    assert all(calls.count(worker) == 2 for worker in WORKERS)
    assert result["control"]["evidence_ready"] is False
    assert result["control"]["step_count"] <= result["control"]["max_steps"]
    assert result["quality_result"]["status"] == "needs_review"
