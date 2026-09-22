from copy import deepcopy
from pathlib import Path

import yaml

from kv_cache_agent.agents import verifier
from kv_cache_agent.agents.verifier import (
    ClaimComparisonBatch,
    ClaimEvidenceDecision,
    evidence_verification_agent,
)

PROMPT_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "kv_cache_agent"
    / "prompts"
    / "verifier.yaml"
)


def _sample_cards() -> list[dict[str, object]]:
    """두 기술에 같은 관점을 적용한 검증 테스트용 EvidenceCard를 만든다."""
    return [
        {
            "evidence_id": "cloud-domain-turboquant-001",
            "technology": "TurboQuant",
            "perspective": "cloud_domain",
            "claim": "TurboQuant는 KV Cache 메모리 사용량을 줄인다.",
            "evidence_text": "TurboQuant reduces KV Cache memory usage.",
            "source_title": "TurboQuant official documentation",
            "source_url": "https://example.com/turboquant",
            "source_type": "official",
            "source_locator": "Performance section",
            "retrieval_method": "tavily",
            "published_date": "2026-01-10",
            "claim_type": "fact",
            "confidence": 0.9,
            "caveat": "특정 평가 환경 기준",
            "verification_status": "unverified",
        },
        {
            "evidence_id": "cloud-domain-cxl_based-002",
            "technology": "CXL-based",
            "perspective": "cloud_domain",
            "claim": "CXL-based 접근은 KV Cache 메모리 용량을 확장한다.",
            "evidence_text": "CXL expands the memory tier for KV Cache.",
            "source_title": "CXL official documentation",
            "source_url": "https://example.com/cxl",
            "source_type": "official",
            "source_locator": "Architecture section",
            "retrieval_method": "tavily",
            "published_date": "2026-01-11",
            "claim_type": "fact",
            "confidence": 0.9,
            "caveat": "CXL 하드웨어 구성이 필요함",
            "verification_status": "unverified",
        },
    ]


def _fake_source(card: dict[str, object]) -> dict[str, object]:
    """실제 네트워크 호출 없이 원문 재확인 성공 결과를 반환한다."""
    return {
        "title": str(card["source_title"]),
        "url": str(card["source_url"]),
        "content": str(card["evidence_text"]),
        "source_type": "web",
        "published_date": str(card["published_date"]),
        "content_length": len(str(card["evidence_text"])),
        "status_code": 200,
        "content_type": "text/html",
        "fetch_status": "ok",
        "error": "",
    }


def _full_support(
    cards: list[dict[str, object]],
    _sources: dict[str, dict[str, object]],
) -> ClaimComparisonBatch:
    """입력된 모든 카드가 원문에 의해 직접 지원된다고 판정한다."""
    return ClaimComparisonBatch(
        decisions=[
            ClaimEvidenceDecision(
                evidence_id=str(card["evidence_id"]),
                support_level="full",
                matched_text=str(card["evidence_text"]),
                rationale="원문이 주장을 직접 뒷받침한다.",
                claim_type_assessment="correct",
            )
            for card in cards
        ]
    )


def _patch_successful_verification(monkeypatch) -> None:
    """원문 조회와 LLM 비교를 Mock 처리하여 외부 API 호출을 차단한다."""
    monkeypatch.setattr(verifier, "_fetch_original_source", _fake_source)
    monkeypatch.setattr(verifier, "_compare_claims_with_sources", _full_support)


def test_verification_graph_contains_fixed_flow_nodes() -> None:
    """도식 B~L 단계가 실제 LangGraph add_node로 등록됐는지 확인한다."""
    graph = verifier.build_evidence_verification_graph().get_graph()
    expected_nodes = {
        "check_source_metadata",
        "recheck_original_sources",
        "compare_claim_and_evidence",
        "evaluate_source_quality",
        "classify_fact_and_inference",
        "check_comparison_balance",
        "check_verification_pass",
        "request_revision",
        "check_reverification_available",
        "mark_uncertainty",
        "return_verified_result",
    }

    assert expected_nodes.issubset(graph.nodes)


def test_verifier_returns_common_result_and_verified_cards(monkeypatch) -> None:
    _patch_successful_verification(monkeypatch)
    original_cards = _sample_cards()
    state = {"evidence_cards": deepcopy(original_cards)}

    result = evidence_verification_agent(state)
    agent_result = result["verification_result"]

    assert set(agent_result) == {
        "agent_name",
        "status",
        "summary",
        "evidence_ids",
        "limitations",
        "errors",
        "payload",
    }
    assert agent_result["agent_name"] == "evidence_verification"
    assert agent_result["status"] == "ok"
    assert len(agent_result["evidence_ids"]) == 2
    assert len(agent_result["payload"]["verified_evidence_cards"]) == 2
    assert agent_result["payload"]["partially_verified_cards"] == []
    assert agent_result["payload"]["unsupported_cards"] == []
    assert agent_result["payload"]["retry_count"] == 0
    assert all(
        card["verification_status"] == "verified"
        for card in agent_result["payload"]["verified_evidence_cards"]
    )
    # 검증기는 GlobalState 입력 카드를 직접 변경하지 않는다.
    assert state["evidence_cards"] == original_cards


def test_verifier_rechecks_failed_card_once(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "_fetch_original_source", _fake_source)
    comparison_calls = 0

    def staged_comparison(
        cards: list[dict[str, object]],
        _sources: dict[str, dict[str, object]],
    ) -> ClaimComparisonBatch:
        nonlocal comparison_calls
        comparison_calls += 1
        decisions = []
        for card in cards:
            is_cxl = card["technology"] == "CXL-based"
            support_level = "partial" if comparison_calls == 1 and is_cxl else "full"
            decisions.append(
                ClaimEvidenceDecision(
                    evidence_id=str(card["evidence_id"]),
                    support_level=support_level,
                    matched_text=str(card["evidence_text"]),
                    rationale="첫 검증은 일부 지원, 재검증은 직접 지원",
                    claim_type_assessment="correct",
                )
            )
        return ClaimComparisonBatch(decisions=decisions)

    monkeypatch.setattr(
        verifier,
        "_compare_claims_with_sources",
        staged_comparison,
    )

    result = evidence_verification_agent({"evidence_cards": _sample_cards()})
    agent_result = result["verification_result"]

    assert comparison_calls == 2
    assert agent_result["status"] == "ok"
    assert agent_result["payload"]["retry_count"] == 1
    assert agent_result["payload"]["retry_requests"]
    assert len(agent_result["payload"]["verified_evidence_cards"]) == 2


def test_verifier_marks_uncertainty_after_retry_limit(monkeypatch) -> None:
    monkeypatch.setattr(verifier, "_fetch_original_source", _fake_source)

    def partial_support(
        cards: list[dict[str, object]],
        _sources: dict[str, dict[str, object]],
    ) -> ClaimComparisonBatch:
        return ClaimComparisonBatch(
            decisions=[
                ClaimEvidenceDecision(
                    evidence_id=str(card["evidence_id"]),
                    support_level="partial",
                    matched_text=str(card["evidence_text"]),
                    rationale="원문이 주장의 일부만 뒷받침한다.",
                    claim_type_assessment="correct",
                )
                for card in cards
            ]
        )

    monkeypatch.setattr(
        verifier,
        "_compare_claims_with_sources",
        partial_support,
    )

    result = evidence_verification_agent({"evidence_cards": _sample_cards()})
    agent_result = result["verification_result"]

    assert agent_result["status"] == "insufficient_evidence"
    assert agent_result["payload"]["retry_count"] == 1
    assert len(agent_result["payload"]["partially_verified_cards"]) == 2
    assert agent_result["payload"]["uncertain_evidence_ids"]
    assert any("불확실한 근거" in item for item in agent_result["limitations"])


def test_verifier_marks_missing_metadata_as_unsupported(monkeypatch) -> None:
    _patch_successful_verification(monkeypatch)
    cards = _sample_cards()
    cards[0]["source_title"] = ""

    result = evidence_verification_agent({"evidence_cards": cards})
    agent_result = result["verification_result"]

    assert agent_result["payload"]["retry_count"] == 1
    assert len(agent_result["payload"]["unsupported_cards"]) == 1
    unsupported = agent_result["payload"]["unsupported_cards"][0]
    assert unsupported["technology"] == "TurboQuant"
    assert "필수 메타데이터 누락" in unsupported["caveat"]


def test_verifier_returns_insufficient_when_cards_are_empty() -> None:
    result = evidence_verification_agent({"evidence_cards": []})

    agent_result = result["verification_result"]
    assert agent_result["status"] == "insufficient_evidence"
    assert agent_result["evidence_ids"] == []
    assert agent_result["payload"]["all_verified_cards"] == []


def test_verifier_prompt_uses_required_yaml_format() -> None:
    prompt = yaml.safe_load(PROMPT_PATH.read_text(encoding="utf-8"))

    assert set(prompt) == {
        "agent_name",
        "version",
        "role",
        "goals",
        "system_prompt",
        "input_fields",
        "output_fields",
        "constraints",
    }
    assert prompt["agent_name"] == "evidence_verification"
    assert prompt["constraints"]["orchestration"] == "langgraph"
    assert prompt["constraints"]["fixed_graph_flow"] is True
    assert prompt["constraints"]["maximum_retries"] == 1
    assert prompt["constraints"]["require_original_source_check"] is True
