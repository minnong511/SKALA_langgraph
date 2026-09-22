from typing import Any

from kv_cache_agent.agents import market
from kv_cache_agent.agents.market import market_evaluation_agent


def _technical_result() -> dict[str, Any]:
    """시장 평가가 사용할 수 있는 최소 기술 조사 결과를 만든다."""
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
    """시장 평가 Agent에 전달할 테스트 State를 만든다."""
    return {
        "user_query": "Compare TurboQuant and CXL-based for cloud LLM serving.",
        "research_plan": {
            "technologies": ["TurboQuant", "CXL-based"],
            "perspectives": ["market"],
            "search_questions": {},
        },
        "technical_result": _technical_result(),
    }


def _search_result(
    *,
    technology: str,
    category: str,
    number: int,
    claim_type: str = "fact",
) -> dict[str, Any]:
    """Tavily의 정규화된 결과와 호환되는 테스트 결과를 만든다."""
    return {
        "title": f"{technology} market source {number}",
        "url": f"https://example.com/{technology.lower()}-{category}-{number}",
        "content": (
            f"{technology} 관련 {category} 자료입니다. "
            "공개 자료에 기반한 시장 정보가 포함되어 있습니다."
        ),
        "score": 0.9,
        "published_date": "2025-01-01",
        "technology": technology,
        "category": category,
        "claim_type": claim_type,
        "source_type": "official",
    }


def test_market_agent_collects_results_and_returns_agent_result(monkeypatch):
    """정상 검색에서 기술별 질문, 카드, AgentResult 구조를 확인한다."""
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
        if "TurboQuant" in query and "CXL-based" not in query:
            technology = "TurboQuant"
            categories = ["market size", "cloud adoption", "commercialization"]
        elif "CXL-based" in query and "TurboQuant" not in query:
            technology = "CXL-based"
            categories = ["competition", "cost", "barriers"]
        else:
            technology = "both"
            categories = ["market size", "cost", "barriers"]

        return [
            _search_result(
                technology=technology,
                category=category,
                number=index,
                claim_type="inference" if category == "cost" else "fact",
            )
            for index, category in enumerate(categories, start=1)
        ]

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    state = _state()
    state["research_plan"]["search_questions"]["market"] = [
        "TurboQuant",
        "CXL-based",
        "TurboQuant CXL-based",
    ]
    result = market_evaluation_agent(state)
    market_result = result["market_result"]
    cards = result["evidence_cards"]

    assert len(calls) == 3
    assert any("TurboQuant" in call["query"] for call in calls)
    assert any("CXL-based" in call["query"] for call in calls)
    assert all(call["max_results"] == 3 for call in calls)
    assert all(call["include_raw_content"] is True for call in calls)

    assert market_result["agent_name"] == "market_evaluation"
    assert market_result["status"] == "ok"
    assert set(market_result) == {
        "agent_name",
        "status",
        "summary",
        "evidence_ids",
        "limitations",
        "errors",
        "payload",
    }
    assert {"queries", "findings", "gaps", "sources"}.issubset(
        market_result["payload"]
    )
    assert {"TurboQuant", "CXL-based", "both"}.issubset(
        {card["technology"] for card in cards}
    )
    assert {"fact", "inference"}.issubset(
        {card["claim_type"] for card in cards}
    )
    assert all(card["retrieval_method"] == "tavily" for card in cards)
    assert all(card["source_url"] for card in cards)
    assert all(card["published_date"] == "2025-01-01" for card in cards)
    assert market_result["evidence_ids"] == [
        card["evidence_id"] for card in cards
    ]


def test_market_agent_supplements_queries_once_when_evidence_is_insufficient(
    monkeypatch,
):
    """근거가 부족하면 보완 검색을 한 번 수행한 뒤 종료하는지 확인한다."""
    calls: list[str] = []

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        calls.append(query)
        return [
            _search_result(
                technology="TurboQuant",
                category="market size",
                number=len(calls),
            )
        ]

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    result = market_evaluation_agent(_state())

    assert len(calls) == 6
    assert any("cloud provider adoption" in query for query in calls)
    assert result["market_result"]["status"] == "insufficient_evidence"
    assert result["market_result"]["payload"]["gaps"]


def test_market_agent_returns_insufficient_evidence_for_empty_results(monkeypatch):
    """검색 결과가 없으면 근거 부족 상태를 반환하는지 확인한다."""

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        return []

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    result = market_evaluation_agent(_state())

    assert result["market_result"]["status"] == "insufficient_evidence"
    assert result["market_result"]["evidence_ids"] == []
    assert result["evidence_cards"] == []
    assert result["market_result"]["payload"]["gaps"]


def test_market_agent_handles_tavily_error(monkeypatch):
    """Tavily 예외가 재시도 가능한 AgentResult로 변환되는지 확인한다."""

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        raise RuntimeError("network failure")

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    result = market_evaluation_agent(_state())

    assert result["market_result"]["status"] == "needs_retry"
    assert result["market_result"]["errors"]
    assert result["evidence_cards"] == []


def test_market_agent_discards_results_without_url_or_content(monkeypatch):
    """URL이나 본문이 없는 검색 결과를 근거로 사용하지 않는지 확인한다."""

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
                "content": "본문은 있지만 URL이 없습니다.",
            },
            {
                "title": "missing content",
                "url": "https://example.com/no-content",
                "content": "",
            },
        ]

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    result = market_evaluation_agent(_state())

    assert result["market_result"]["status"] == "insufficient_evidence"
    assert result["evidence_cards"] == []


def test_market_agent_removes_duplicate_urls(monkeypatch):
    """여러 검색 결과에 같은 URL이 있으면 한 번만 저장하는지 확인한다."""
    calls = 0

    def fake_search_web(
        query: str,
        *,
        max_results: int,
        include_raw_content: bool,
    ) -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return [
            {
                "title": "duplicate source",
                "url": "https://example.com/duplicate",
                "content": "TurboQuant와 CXL-based 시장 정보입니다.",
                "score": 0.8,
                "published_date": "2025-02-01",
            }
        ]

    monkeypatch.setattr(market.tavily_tool, "search_web", fake_search_web)

    result = market_evaluation_agent(_state())

    assert calls <= 6
    assert len(result["evidence_cards"]) == 1
    assert result["market_result"]["evidence_ids"] == [
        result["evidence_cards"][0]["evidence_id"]
    ]


def test_market_agent_returns_failed_when_technical_result_is_missing():
    """기술 조사 결과가 없으면 시장 평가를 시작하지 않는지 확인한다."""
    result = market_evaluation_agent(
        {
            "user_query": "Compare TurboQuant and CXL-based.",
            "research_plan": {"search_questions": {}},
        }
    )

    assert result["market_result"]["status"] == "failed"
    assert result["market_result"]["errors"]
    assert result["evidence_cards"] == []
