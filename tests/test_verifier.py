from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import yaml

from kv_cache_agent.agents import verifier
from kv_cache_agent.agents.verifier import (
    ClaimComparisonBatch,
    ClaimEvidenceDecision,
    evidence_verification_agent,
)

PROMPT_PATH = (
    Path(__file__).parents[1] / "src" / "kv_cache_agent" / "prompts" / "verifier.yaml"
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


def _blocked_source(card: dict[str, object]) -> dict[str, object]:
    """원문 서버가 403을 반환한 상황을 재현한다."""
    return {
        "title": str(card["source_title"]),
        "url": str(card["source_url"]),
        "content": "",
        "source_type": "web",
        "published_date": str(card["published_date"]),
        "content_length": 0,
        "status_code": 403,
        "content_type": "text/html",
        "fetch_status": "blocked",
        "error": "원문 접근이 차단되었습니다: HTTP 403",
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
    assert (
        result["verified_evidence_cards"]
        == agent_result["payload"]["verified_evidence_cards"]
    )
    assert (
        result["usable_evidence_cards"]
        == agent_result["payload"]["verified_evidence_cards"]
    )
    assert agent_result["payload"]["partially_verified_cards"] == []
    assert agent_result["payload"]["unsupported_cards"] == []
    assert agent_result["payload"]["retry_count"] == 0
    assert all(
        card["verification_status"] == "verified"
        for card in agent_result["payload"]["verified_evidence_cards"]
    )
    # 검증기는 GlobalState 입력 카드를 직접 변경하지 않는다.
    assert state["evidence_cards"] == original_cards


def test_verifier_uses_tavily_excerpt_as_partial_fallback(monkeypatch) -> None:
    """원문 차단 시 Tavily 요약을 사용하되 검증 완료로 승격하지 않는다."""
    monkeypatch.setattr(verifier, "_fetch_original_source", _blocked_source)
    monkeypatch.setattr(verifier, "_compare_claims_with_sources", _full_support)

    result = evidence_verification_agent({"evidence_cards": _sample_cards()})
    agent_result = result["verification_result"]

    assert agent_result["status"] == "insufficient_evidence"
    assert len(agent_result["payload"]["partially_verified_cards"]) == 2
    assert agent_result["payload"]["verified_evidence_cards"] == []
    assert agent_result["payload"]["tavily_fallback_ids"] == [
        "cloud-domain-turboquant-001",
        "cloud-domain-cxl_based-002",
    ]
    assert all(
        card["verification_status"] == "partially_verified"
        for card in agent_result["payload"]["partially_verified_cards"]
    )
    assert all(
        "원문 접근 차단" in card["caveat"]
        for card in agent_result["payload"]["partially_verified_cards"]
    )
    assert (
        result["usable_evidence_cards"]
        == agent_result["payload"]["partially_verified_cards"]
    )
    assert any("Tavily 검색 요약" in item for item in agent_result["limitations"])


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
    assert result["verified_evidence_cards"] == []
    assert result["usable_evidence_cards"] == []


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


def test_verifier_fetches_distinct_pdf_locators_separately(monkeypatch):
    cards = _sample_cards()
    for card, page in zip(cards, ("p. 1", "p. 9"), strict=True):
        card.update(
            source_url="data/papers/shared.pdf",
            source_type="paper",
            source_locator=page,
        )
    calls = []

    def source(card):
        calls.append(card["source_locator"])
        return {**_fake_source(card), "content": card["source_locator"]}

    def compare(cards, sources):
        assert all(
            sources[c["evidence_id"]]["content"] == c["source_locator"] for c in cards
        )
        return _full_support(cards, sources)

    monkeypatch.setattr(verifier, "_fetch_original_source", source)
    monkeypatch.setattr(verifier, "_compare_claims_with_sources", compare)
    result = evidence_verification_agent({"evidence_cards": cards})
    assert calls == ["p. 1", "p. 9"]
    assert result["verification_result"]["status"] == "ok"


def test_pdf_locator_selects_sparse_pages_and_ignores_table_numbers(monkeypatch):
    reader = Mock()
    reader.pages = [Mock() for _ in range(10)]
    for index, page in enumerate(reader.pages, 1):
        page.extract_text.return_value = f"unique-page-{index}"
    monkeypatch.setattr(verifier, "PdfReader", lambda _: reader)
    text = verifier._extract_pdf_pages(Path("paper.pdf"), "p. 2; p. 10; Table 4")
    assert "unique-page-2" in text and "unique-page-10" in text
    assert "unique-page-4" not in text and "unique-page-9" not in text
    text = verifier._extract_pdf_pages(Path("paper.pdf"), "pp. 2-4")
    assert all(f"unique-page-{i}" in text for i in (2, 3, 4))
    assert "unique-page-5" not in text


def test_source_excerpt_includes_numeric_evidence_beyond_prefix():
    text = (
        "intro " * 3000
        + "Compared with NVMe-oF, achieves a 1 .80× throughput improvement."
    )
    card = {
        "claim": "NVMe-oF 대비 1.80배",
        "evidence_text": "1.80× throughput improvement",
    }
    excerpt = verifier._source_excerpt(card, text)
    assert "1 .80× throughput improvement" in excerpt
    assert len(excerpt) <= verifier.MAX_SOURCE_CHARS_PER_CARD


def test_source_excerpt_includes_integer_batch_and_latency_metrics():
    text = (
        "intro " * 3000
        + "For PC-CXL the batch size is 57. The latency ranges from 55 to 336ms."
    )
    card = {
        "claim": "최대 배치 크기 57, TTFT 55–336ms",
        "evidence_text": "30% increase",
    }
    excerpt = verifier._source_excerpt(card, text)
    assert "batch size is 57" in excerpt
    assert "55 to 336ms" in excerpt


def test_comparisons_isolate_sources_and_reject_cross_source_quotes(monkeypatch):
    cards = _sample_cards()
    cards[0]["evidence_text"] = "NVMe-oF throughput improves by 1.80×."
    cards[1]["evidence_text"] = "CPU offloading throughput improves by 35.7%."
    sources = {str(c["evidence_id"]): _fake_source(c) for c in cards}
    # PDF typography differs, but the literal quote should still be recognized.
    sources[str(cards[0]["evidence_id"])]["content"] = (
        "NVMe-oF throughput improves by 1 .80×."
    )
    batch = _full_support(cards, sources)
    batch.decisions[1].matched_text = str(cards[0]["evidence_text"])
    llm = Mock()
    llm.with_structured_output.return_value.batch.return_value = [
        ClaimComparisonBatch(decisions=[d]) for d in batch.decisions
    ]
    monkeypatch.setattr(verifier, "get_llm", lambda: llm)
    result = verifier._compare_claims_with_sources(cards, sources)
    inputs = llm.with_structured_output.return_value.batch.call_args.args[0]
    assert "35.7%" not in inputs[0][1].content
    assert "NVMe-oF" not in inputs[1][1].content
    assert result.decisions[0].support_level == "full"
    assert result.decisions[1].support_level == "none"
    assert result.decisions[1].matched_text == ""


def test_single_comparison_exception_keeps_other_card_results(monkeypatch):
    cards = _sample_cards()
    sources = {str(c["evidence_id"]): _fake_source(c) for c in cards}
    llm = Mock()
    llm.with_structured_output.return_value.batch.return_value = [
        _full_support(cards[:1], sources),
        RuntimeError("SECRET provider body"),
    ]
    monkeypatch.setattr(verifier, "get_llm", lambda: llm)
    result = verifier._compare_claims_with_sources(cards, sources)
    assert [d.support_level for d in result.decisions] == ["full", "none"]
    assert "SECRET" not in result.model_dump_json()


def test_multiple_quotes_must_all_belong_to_the_same_source(monkeypatch):
    cards = _sample_cards()[:1]
    sources = {
        str(cards[0]["evidence_id"]): {
            **_fake_source(cards[0]),
            "content": "Prefetching overlaps data transfer. Intervening text. NVMe-oF throughput improves by 1.80×.",
        }
    }
    decision = _full_support(cards, sources).decisions[0]
    decision.matched_quotes = [
        "Prefetching overlaps data transfer.",
        "NVMe-oF throughput improves by 1.80×.",
    ]
    llm = Mock()
    llm.with_structured_output.return_value.batch.return_value = [
        ClaimComparisonBatch(decisions=[decision])
    ]
    monkeypatch.setattr(verifier, "get_llm", lambda: llm)
    assert (
        verifier._compare_claims_with_sources(cards, sources).decisions[0].support_level
        == "full"
    )
    decision.matched_quotes[1] = "CPU-offload throughput improves by 35.7%."
    assert (
        verifier._compare_claims_with_sources(cards, sources).decisions[0].support_level
        == "none"
    )


def test_retrieval_absence_is_a_limitation_and_cannot_be_fact_evidence(monkeypatch):
    card = _sample_cards()[1]
    card["claim"] = (
        "검색된 발췌만으로는 NVMe-oF 대비 1.80배 처리량 주장을 검증할 수 없다."
    )
    llm = Mock()
    monkeypatch.setattr(verifier, "get_llm", lambda: llm)
    result = verifier._compare_claims_with_sources(
        [card], {str(card["evidence_id"]): _fake_source(card)}
    )
    llm.with_structured_output.return_value.batch.assert_not_called()
    assert result.decisions[0].support_level == "none"
    update = verifier._compare_claim_and_evidence_node(
        {
            "cards": [card],
            "active_evidence_ids": [str(card["evidence_id"])],
            "source_documents": {str(card["evidence_id"]): _fake_source(card)},
        }
    )
    assert card["claim"] in update["limitations"][0]
