from typing import Any

from kv_cache_agent.agents import stakeholder
from kv_cache_agent.agents.stakeholder import stakeholder_evaluation_agent


def _technical_result() -> dict[str, Any]:
    """이해관계자 평가가 사용할 수 있는 최소 기술 조사 결과를 만든다."""
    return {
        "agent_name": "technical_research",
        "status": "ok",
        "summary": "TurboQuant와 CXL-based 기술 조사 결과",
        "evidence_ids": ["technical-001"],
        "limitations": [],
        "errors": [],
        "payload": {
            "technologies": ["TurboQuant", "CXL-based"],
        },
    }


def _state() -> dict[str, Any]:
    """이해관계자 평가 Agent에 전달할 테스트 State를 만든다."""
    return {
        "user_query": "Compare TurboQuant and CXL-based for cloud LLM serving.",
        "research_plan": {
            "technologies": ["TurboQuant", "CXL-based"],
            "perspectives": ["stakeholder"],
            "search_questions": {},
        },
        "technical_result": _technical_result(),
    }


def _search_result(
    *,
    number: int,
    technology: str,
    stakeholders: list[str],
    claim_type: str = "fact",
) -> dict[str, Any]:
    """Tavily 정규화 결과와 호환되는 목업 검색 결과를 만든다."""
    return {
        "title": f"{technology} stakeholder source {number}",
        "url": f"https://example.com/stakeholder-{number}",
        "content": (
            "The expected benefit is lower serving cost and broader adoption, "
            "but integration concern and operational risk remain."
        ),
        "score": 0.9,
        "published_date": "2025-01-01",
        "technology": technology,
        "stakeholders": stakeholders,
        "claim_type": claim_type,
        "source_type": "official",
    }


def test_stakeholder_agent_collects_six_views_and_returns_agent_result(monkeypatch):
    """정상 검색에서 6개 이해관계자와 기대·우려 근거를 반환하는지 확인한다."""
    calls: list[dict[str, Any]] = []

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "query": query,
                "max_results": max_results,
                "include_raw_content": include_raw_content,
            }
        )
        groups = [
            ["클라우드 사업자", "클라우드 고객"],
            ["AI 모델 개발사", "오픈소스 개발자"],
            ["하드웨어 제조사", "연구자·투자자"],
        ]
        technologies = ["TurboQuant", "CXL-based", "both"]
        index = len(calls) - 1
        return [
            _search_result(
                number=index + 1,
                technology=technologies[index],
                stakeholders=groups[index],
                claim_type="inference" if index == 1 else "fact",
            )
        ]

    monkeypatch.setattr(stakeholder.tavily_tool, "search_web", fake_search_web)

    state = _state()
    state["research_plan"]["search_questions"]["stakeholder"] = [
        "cloud stakeholder",
        "model stakeholder",
        "hardware stakeholder",
    ]
    result = stakeholder_evaluation_agent(state)
    stakeholder_result = result["stakeholder_result"]
    cards = result["evidence_cards"]

    assert len(calls) == 3
    assert all(call["max_results"] == 3 for call in calls)
    assert all(call["include_raw_content"] is True for call in calls)
    assert stakeholder_result["agent_name"] == "stakeholder_evaluation"
    assert stakeholder_result["status"] == "ok"
    assert set(stakeholder_result) == {
        "agent_name",
        "status",
        "summary",
        "evidence_ids",
        "limitations",
        "errors",
        "payload",
    }
    findings = stakeholder_result["payload"]["findings"]
    assert {
        "클라우드 사업자",
        "AI 모델 개발사",
        "하드웨어 제조사",
        "클라우드 고객",
        "오픈소스 개발자",
        "연구자·투자자",
    } == {finding["stakeholder"] for finding in findings}
    assert all(finding["expectations"] for finding in findings)
    assert all(finding["concerns"] for finding in findings)
    assert {"fact", "inference"}.issubset(
        {card["claim_type"] for card in cards}
    )
    assert all(card["perspective"] == "stakeholder" for card in cards)
    assert all(card["retrieval_method"] == "tavily" for card in cards)
    assert all(card["source_url"] for card in cards)
    assert all(card["published_date"] == "2025-01-01" for card in cards)
    assert stakeholder_result["evidence_ids"] == [
        card["evidence_id"] for card in cards
    ]


def test_stakeholder_agent_retries_only_missing_stakeholder(monkeypatch):
    """근거가 빠진 유형만 한 번 보완 검색하고 정상 종료하는지 확인한다."""
    calls: list[str] = []
    first_group = [
        "클라우드 사업자",
        "AI 모델 개발사",
        "하드웨어 제조사",
        "클라우드 고객",
        "오픈소스 개발자",
    ]

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        calls.append(query)
        if len(calls) <= 3:
            return [
                _search_result(
                    number=1,
                    technology="TurboQuant",
                    stakeholders=first_group,
                )
            ]
        return [
            _search_result(
                number=2,
                technology="CXL-based",
                stakeholders=["연구자·투자자"],
                claim_type="inference",
            )
        ]

    monkeypatch.setattr(stakeholder.tavily_tool, "search_web", fake_search_web)

    result = stakeholder_evaluation_agent(_state())

    assert len(calls) == 4
    assert "researchers investors" in calls[-1]
    assert result["stakeholder_result"]["status"] == "ok"
    assert result["stakeholder_result"]["payload"]["gaps"] == []


def test_stakeholder_agent_marks_insufficient_evidence_for_empty_results(monkeypatch):
    """검색 결과가 없으면 재시도 후 근거 부족 상태를 반환하는지 확인한다."""
    calls = 0

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return []

    monkeypatch.setattr(stakeholder.tavily_tool, "search_web", fake_search_web)

    result = stakeholder_evaluation_agent(_state())

    assert calls == 6
    assert result["stakeholder_result"]["status"] == "insufficient_evidence"
    assert result["stakeholder_result"]["evidence_ids"] == []
    assert result["evidence_cards"] == []
    assert set(result["stakeholder_result"]["payload"]["gaps"]) == set(
        stakeholder.STAKEHOLDERS
    )
    uncertain_findings = result["stakeholder_result"]["payload"]["findings"]
    assert all(finding["position"] == "uncertain" for finding in uncertain_findings)
    assert all(finding["claim_type"] == "inference" for finding in uncertain_findings)


def test_stakeholder_agent_handles_tavily_error(monkeypatch):
    """Tavily 예외를 재시도 가능한 AgentResult로 변환하는지 확인한다."""

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("network failure")

    monkeypatch.setattr(stakeholder.tavily_tool, "search_web", fake_search_web)

    result = stakeholder_evaluation_agent(_state())

    assert result["stakeholder_result"]["status"] == "needs_retry"
    assert result["stakeholder_result"]["errors"]
    assert result["evidence_cards"] == []


def test_stakeholder_agent_discards_results_without_source(monkeypatch):
    """URL이나 본문 또는 이해관계자 유형이 없는 결과를 버리는지 확인한다."""

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        return [
            {
                "title": "missing url",
                "url": "",
                "content": "cloud provider benefit",
            },
            {
                "title": "missing content",
                "url": "https://example.com/no-content",
                "content": "",
            },
            {
                "title": "no stakeholder",
                "url": "https://example.com/no-stakeholder",
                "content": "unrelated technical note",
            },
        ]

    monkeypatch.setattr(stakeholder.tavily_tool, "search_web", fake_search_web)

    result = stakeholder_evaluation_agent(_state())

    assert result["stakeholder_result"]["status"] == "insufficient_evidence"
    assert result["evidence_cards"] == []


def test_stakeholder_agent_returns_failed_when_technical_result_is_missing():
    """기술 조사 결과가 없으면 검색을 시작하지 않는지 확인한다."""
    result = stakeholder_evaluation_agent(
        {
            "user_query": "Compare TurboQuant and CXL-based.",
            "research_plan": {"search_questions": {}},
        }
    )

    assert result["stakeholder_result"]["status"] == "failed"
    assert result["stakeholder_result"]["errors"]
    assert result["evidence_cards"] == []
