"""Integration tests use real agents/graph and mock only external services."""

from kv_cache_agent.config import WorkflowLimits
from kv_cache_agent.graph.workflow import build_workflow
from kv_cache_agent.mock_run import mock_services


def test_full_workflow_reaches_report_writer_and_quality_evaluator():
    with mock_services():
        result = build_workflow(limits=WorkflowLimits()).invoke(
            {
                "payload": {"user_query": "두 기술의 클라우드 LLM 서빙을 평가해줘"},
            }
        )
    assert result["control"]["status"] == "completed"
    assert result["control"]["termination_reason"] == "quality_pass"
    assert result["payload"]["evaluation"]["overall_pass"] is True
    assert result["payload"]["report"].startswith("# SUMMARY")
    assert result["payload"]["verified_evidence_cards"]
    assert set(result) == {"payload", "control"}
    assert len(result["payload"]["tasks"]) == len(result["payload"]["worker_results"])
