import json
from pathlib import Path

import yaml

from kv_cache_agent.agents import cloud_domain
from kv_cache_agent.agents.cloud_domain import (
    CLOUD_EVALUATION_CRITERIA,
    CloudComparison,
    CloudDomainExtraction,
    CloudDomainFinding,
    cloud_domain_agent,
)
from kv_cache_agent.tools.tavily_search import TavilySearchError

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "sample_technical_result.json"
PROMPT_PATH = (
    Path(__file__).parents[1]
    / "src"
    / "kv_cache_agent"
    / "prompts"
    / "cloud_domain.yaml"
)


def _sample_state() -> dict[str, object]:
    """기술 조사 fixture를 실제 GlobalState 입력과 같은 모양으로 만든다."""
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    return {
        "user_query": "두 기술을 클라우드 LLM 서빙 관점에서 비교해 줘.",
        "technical_result": fixture["technical_result"],
        "evidence_cards": fixture["evidence_cards"],
    }


def _fake_web_results(*_: object, **__: object) -> list[dict[str, object]]:
    """실제 Tavily를 호출하지 않고 두 기술의 정규화된 검색 결과를 반환한다."""
    return [
        {
            "result_id": "web-turboquant",
            "title": "TurboQuant cloud serving analysis",
            "url": "https://example.com/turboquant-cloud",
            "content": "TurboQuant reduces KV cache memory use in serving.",
            "score": 0.9,
            "raw_content": None,
            "published_date": "2026-01-10",
            "favicon": "",
            "topic": "general",
            "source_type": "web",
            "retrieval_method": "tavily",
        },
        {
            "result_id": "web-cxl",
            "title": "CXL KV cache cloud serving analysis",
            "url": "https://example.com/cxl-cloud",
            "content": "CXL expands the memory tier for long-context serving.",
            "score": 0.88,
            "raw_content": None,
            "published_date": "2026-01-11",
            "favicon": "",
            "topic": "general",
            "source_type": "web",
            "retrieval_method": "tavily",
        },
    ]


def _fake_vector_chunks(*_: object, **__: object) -> list[dict[str, object]]:
    """실제 FAISS를 조회하지 않고 논문 Vector DB 결과 형식을 반환한다."""
    return [
        {
            "chunk_id": "turboquant:p1:c0",
            "content": "TurboQuant paper evidence",
            "metadata": {
                "source_title": "TurboQuant",
                "source_path": "data/papers/turboquant.pdf",
                "source_locator": "p. 1",
            },
        },
        {
            "chunk_id": "cxl:p1:c0",
            "content": "CXL paper evidence",
            "metadata": {
                "source_title": "ITME",
                "source_path": "data/papers/cxl_based_kv_cache.pdf",
                "source_locator": "p. 1",
            },
        },
    ]


def _extraction_for(
    coverage: list[tuple[str, str]],
) -> CloudDomainExtraction:
    """지정한 기술·시나리오 조합을 모두 평가한 가짜 LLM 결과를 만든다."""
    findings = []
    for technology, criterion in coverage:
        source_url = (
            "https://example.com/turboquant-cloud"
            if technology == "TurboQuant"
            else "https://example.com/cxl-cloud"
        )
        findings.append(
            CloudDomainFinding(
                technology=technology,
                criterion=criterion,
                claim=f"{technology}의 {criterion} 클라우드 평가 주장",
                evidence_text=f"{technology}의 {criterion} 관련 검색 근거",
                source_url=source_url,
                source_type="report",
                claim_type="inference",
                confidence=0.8,
                caveat="실제 클라우드 환경에서 추가 검증이 필요하다.",
            )
        )
    return CloudDomainExtraction(
        summary="두 기술의 주요 클라우드 시나리오를 같은 기준으로 평가했다.",
        comparison=CloudComparison(
            turboquant="소프트웨어 기반 KV Cache 압축",
            cxl_based="하드웨어 기반 원격 메모리 확장",
            trade_off="메모리 절감과 용량 확장의 차이가 있다.",
        ),
        findings=findings,
        limitations=[],
    )


def _complete_extraction(*_: object, **__: object) -> CloudDomainExtraction:
    """7개 시나리오와 두 기술의 총 14개 조합을 모두 평가한다."""
    coverage = [
        (technology, criterion)
        for technology in ("TurboQuant", "CXL-based")
        for criterion in CLOUD_EVALUATION_CRITERIA
    ]
    return _extraction_for(coverage)


def _patch_successful_retrieval(monkeypatch) -> None:
    """단위 테스트에서 Tavily와 FAISS 외부 의존성을 모두 Mock 처리한다."""
    monkeypatch.setattr(cloud_domain, "search_web", _fake_web_results)
    monkeypatch.setattr(
        cloud_domain,
        "retrieve_paper_chunks",
        _fake_vector_chunks,
    )


def test_cloud_domain_agent_follows_scenarios_and_common_result_format(
    monkeypatch,
) -> None:
    _patch_successful_retrieval(monkeypatch)
    monkeypatch.setattr(
        cloud_domain,
        "_extract_cloud_assessment",
        _complete_extraction,
    )

    result = cloud_domain_agent(_sample_state())
    agent_result = result["cloud_domain_result"]

    assert set(agent_result) == {
        "agent_name",
        "status",
        "summary",
        "evidence_ids",
        "limitations",
        "errors",
        "payload",
    }
    assert agent_result["agent_name"] == "cloud_domain_evaluation"
    assert agent_result["status"] == "ok"
    assert len(agent_result["payload"]["scenarios"]) == 7
    assert agent_result["payload"]["missing_scenarios"] == []
    assert agent_result["payload"]["retry_count"] == 0
    assert agent_result["payload"]["vector_source_count"] == 2
    assert len(result["evidence_cards"]) == 14

    required_card_fields = {
        "evidence_id",
        "technology",
        "perspective",
        "claim",
        "evidence_text",
        "source_title",
        "source_url",
        "source_type",
        "source_locator",
        "retrieval_method",
        "published_date",
        "claim_type",
        "confidence",
        "caveat",
        "verification_status",
    }
    for card in result["evidence_cards"]:
        assert set(card) == required_card_fields
        assert card["perspective"] == "cloud_domain"
        assert card["verification_status"] == "unverified"
        assert card["claim_type"] == "inference"


def test_cloud_domain_graph_contains_fixed_flow_nodes() -> None:
    """B~K 단계가 실제 LangGraph add_node로 등록됐는지 확인한다."""
    graph = cloud_domain.build_cloud_domain_graph().get_graph()
    expected_nodes = {
        "generate_scenarios",
        "generate_search_queries",
        "retrieve_sources",
        "extract_related_evidence",
        "evaluate_domain_fit",
        "check_scenario_coverage",
        "prepare_retry",
        "check_retry_available",
        "record_limitations",
        "return_domain_result",
    }

    assert expected_nodes.issubset(graph.nodes)


def test_cloud_domain_agent_researches_missing_scenarios_once(monkeypatch) -> None:
    searched_queries: list[str] = []
    extraction_calls = 0

    def fake_search(query: str, **kwargs: object) -> list[dict[str, object]]:
        del kwargs
        searched_queries.append(query)
        return _fake_web_results()

    def staged_extraction(*_: object, **kwargs: object) -> CloudDomainExtraction:
        nonlocal extraction_calls
        extraction_calls += 1
        if extraction_calls == 1:
            return _extraction_for(
                [
                    ("TurboQuant", "gpu_memory_cost"),
                    ("CXL-based", "long_context"),
                ]
            )
        return _extraction_for(kwargs["required_coverage"])

    monkeypatch.setattr(cloud_domain, "search_web", fake_search)
    monkeypatch.setattr(
        cloud_domain,
        "retrieve_paper_chunks",
        _fake_vector_chunks,
    )
    monkeypatch.setattr(
        cloud_domain,
        "_extract_cloud_assessment",
        staged_extraction,
    )
    state = _sample_state()
    state["research_plan"] = {
        "search_questions": {
            "cloud_domain": ["q1", "q2", "q3", "q4"],
        }
    }

    result = cloud_domain_agent(state)
    agent_result = result["cloud_domain_result"]

    assert len(searched_queries) == 3
    assert searched_queries[:2] == ["q1", "q2"]
    assert agent_result["payload"]["retry_count"] == 1
    assert len(agent_result["payload"]["retry_queries"]) == 1
    assert agent_result["payload"]["missing_scenarios"] == []
    assert agent_result["status"] == "ok"


def test_cloud_domain_agent_records_limit_after_retry(monkeypatch) -> None:
    _patch_successful_retrieval(monkeypatch)
    partial = _extraction_for(
        [
            ("TurboQuant", "gpu_memory_cost"),
            ("CXL-based", "long_context"),
        ]
    )
    monkeypatch.setattr(
        cloud_domain,
        "_extract_cloud_assessment",
        lambda *_args, **_kwargs: partial,
    )

    result = cloud_domain_agent(_sample_state())
    agent_result = result["cloud_domain_result"]

    assert agent_result["status"] == "insufficient_evidence"
    assert agent_result["payload"]["retry_count"] == 1
    assert agent_result["payload"]["missing_scenarios"]
    assert any("최대 1회 재검색" in item for item in agent_result["limitations"])


def test_cloud_domain_agent_reports_missing_technical_evidence(monkeypatch) -> None:
    def should_not_search(*_: object, **__: object) -> list[dict[str, object]]:
        raise AssertionError("기술 근거가 없으면 검색하면 안 됩니다.")

    monkeypatch.setattr(cloud_domain, "search_web", should_not_search)
    monkeypatch.setattr(cloud_domain, "retrieve_paper_chunks", should_not_search)

    result = cloud_domain_agent(
        {
            "user_query": "클라우드 적합성을 비교해 줘.",
            "evidence_cards": [],
        }
    )

    agent_result = result["cloud_domain_result"]
    assert agent_result["status"] == "insufficient_evidence"
    assert agent_result["evidence_ids"] == []
    assert agent_result["errors"] == []
    assert result["evidence_cards"] == []


def test_cloud_domain_agent_records_limit_when_searches_fail(
    monkeypatch,
) -> None:
    def failing_search(*_: object, **__: object) -> list[dict[str, object]]:
        raise TavilySearchError("테스트용 Tavily 실패")

    def failing_vector(*_: object, **__: object) -> list[dict[str, object]]:
        raise FileNotFoundError("테스트용 Vector DB 실패")

    monkeypatch.setattr(cloud_domain, "search_web", failing_search)
    monkeypatch.setattr(cloud_domain, "retrieve_paper_chunks", failing_vector)

    result = cloud_domain_agent(_sample_state())

    agent_result = result["cloud_domain_result"]
    assert agent_result["status"] == "insufficient_evidence"
    assert agent_result["payload"]["retry_count"] == 1
    assert agent_result["errors"]
    assert any("최대 1회 재검색" in item for item in agent_result["limitations"])
    assert result["evidence_cards"] == []


def test_cloud_domain_prompt_uses_required_yaml_format() -> None:
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
    assert prompt["agent_name"] == "cloud_domain_evaluation"
    assert prompt["constraints"]["orchestration"] == "langgraph"
    assert prompt["constraints"]["fixed_graph_flow"] is True
    assert prompt["constraints"]["maximum_initial_search_queries"] == 2
    assert prompt["constraints"]["maximum_search_queries"] == 3
    assert prompt["constraints"]["maximum_retries"] == 1
    assert prompt["constraints"]["require_vector_db_search"] is True
    assert prompt["constraints"]["require_citation"] is True
