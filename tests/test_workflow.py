from kv_cache_agent.graph.workflow import build_workflow


def test_full_workflow_reaches_report_writer() -> None:
    def fake_technical_node(state: dict[str, object]) -> dict[str, object]:
        return {
            "technical_result": {
                "agent_name": "technical_research",
                "status": "insufficient_evidence",
                "summary": "테스트용 기술 조사 결과",
                "evidence_ids": [],
                "limitations": [],
                "errors": [],
                "payload": {},
            },
            "evidence_cards": [],
        }

    workflow = build_workflow(technical_node=fake_technical_node)

    result = workflow.invoke(
        {"user_query": "Compare TurboQuant and CXL-based KV Cache for cloud serving."}
    )

    assert result["technical_result"]["status"] == "insufficient_evidence"
    assert result["market_result"]["status"] == "insufficient_evidence"
    assert result["stakeholder_result"]["status"] == "insufficient_evidence"
    assert result["cloud_domain_result"]["status"] == "insufficient_evidence"
    assert result["verification_result"]["status"] == "ok"
    assert "final_report" in result
