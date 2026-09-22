"""Tavily 검색 결과를 시장 평가 근거로 변환하는 에이전트."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict
from urllib.parse import urlparse

from langgraph.graph import END, START, StateGraph

from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.tools import tavily_search as tavily_tool


MAX_TAVILY_QUERIES = 3
MAX_RESULTS_PER_QUERY = 3

TECHNOLOGIES = ("TurboQuant", "CXL-based")

DEFAULT_MARKET_QUERIES = [
    "TurboQuant KV cache compression market size adoption cloud LLM inference",
    "CXL-based KV cache memory expansion market adoption cloud LLM inference",
    "TurboQuant CXL-based KV cache cost competition barriers cloud serving",
]

MARKET_CATEGORIES = {
    "market_size_growth": (
        "market size",
        "market growth",
        "growth",
        "forecast",
        "cagr",
        "시장 규모",
        "성장 전망",
    ),
    "cloud_adoption": (
        "cloud",
        "cloud provider",
        "hyperscaler",
        "adoption",
        "클라우드",
        "채택",
    ),
    "commercialization": (
        "commercial",
        "deployment",
        "product launch",
        "production",
        "상용화",
        "도입 사례",
    ),
    "competition": (
        "competitor",
        "competition",
        "alternative",
        "vendor",
        "경쟁 기술",
        "관련 기업",
    ),
    "cost": (
        "cost",
        "saving",
        "economic",
        "tco",
        "비용",
        "절감",
    ),
    "barriers": (
        "barrier",
        "challenge",
        "limitation",
        "integration",
        "도입 장벽",
        "한계",
    ),
}

MARKET_CATEGORY_LABELS = {
    "market_size_growth": "시장 규모와 성장 전망",
    "cloud_adoption": "클라우드 사업자 채택 가능성",
    "commercialization": "상용화 및 채택 사례",
    "competition": "관련 기업과 경쟁 기술",
    "cost": "비용 절감 가능성",
    "barriers": "도입 장벽",
}

CATEGORY_QUERY_TEMPLATES = {
    "market_size_growth": "{technology} market size growth forecast cloud LLM serving",
    "cloud_adoption": "{technology} cloud provider adoption LLM inference",
    "commercialization": "{technology} commercial deployment adoption case",
    "competition": "{technology} competing technology vendors market",
    "cost": "{technology} cost saving total cost of ownership LLM serving",
    "barriers": "{technology} deployment barrier integration challenge",
}

VALID_CLAIM_TYPES = {"fact", "inference"}
VALID_SOURCE_TYPES = {"paper", "official", "news", "blog", "report"}
TAVILY_FUNCTION_NAMES = ("search_web", "search_tavily", "tavily_search", "search")


class MarketLocalState(TypedDict, total=False):
    """시장 평가 Agent 내부 그래프에서만 사용하는 임시 State."""

    global_state: GlobalState
    queries: list[str]
    current_queries: list[str]
    normalized_results: list[dict[str, Any]]
    source_records: list[dict[str, Any]]
    source_urls: list[str]
    evidence_cards: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    missing_categories: list[str]
    retry_count: int
    max_retries: int
    sufficient: bool
    terminal_status: str
    terminal_summary: str
    terminal_error: str
    market_result: dict[str, Any]


class _TavilyInterfaceError(RuntimeError):
    """Tavily 도구에서 호출 가능한 공개 함수를 찾지 못했을 때 발생한다."""


def _build_queries(state: GlobalState) -> list[str]:
    """ResearchPlan의 시장 질문을 사용하고 부족하면 기본 질문을 보완한다."""
    research_plan = state.get("research_plan", {})
    search_questions = research_plan.get("search_questions", {})
    planned_queries = search_questions.get("market", [])

    queries: list[str] = []
    for query in planned_queries:
        normalized_query = str(query).strip()
        if normalized_query and normalized_query not in queries:
            queries.append(normalized_query)

    for query in DEFAULT_MARKET_QUERIES:
        if len(queries) >= MAX_TAVILY_QUERIES:
            break
        if query not in queries:
            queries.append(query)

    return queries[:MAX_TAVILY_QUERIES]


def _resolve_tavily_function() -> Any:
    """Tool 모듈에서 담당자가 공개한 Tavily 검색 함수를 찾는다."""
    for function_name in TAVILY_FUNCTION_NAMES:
        function = getattr(tavily_tool, function_name, None)
        if callable(function):
            return function
    raise _TavilyInterfaceError(
        "tavily_search.py에 호출 가능한 공개 검색 함수가 없습니다."
    )


def _call_tavily(query: str) -> list[Any]:
    """기존 Tavily 도구를 호출하고 결과 목록으로 정규화한다."""
    search_function = _resolve_tavily_function()

    try:
        response = search_function(
            query=query,
            max_results=MAX_RESULTS_PER_QUERY,
            include_raw_content=True,
        )
    except TypeError as keyword_error:
        # 담당 Tool이 위치 인자만 받는 경우에도 공개 인터페이스를 호출한다.
        try:
            response = search_function(query, MAX_RESULTS_PER_QUERY)
        except TypeError:
            raise keyword_error

    if response is None:
        return []
    if isinstance(response, Mapping):
        for key in ("results", "data"):
            value = response.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return list(value)
        return [response]
    if isinstance(response, Sequence) and not isinstance(response, (str, bytes)):
        return list(response)
    return []


def _as_mapping(value: Any) -> Mapping[str, Any] | None:
    """검색 결과 객체를 딕셔너리 형태로 읽는다."""
    if isinstance(value, Mapping):
        return value
    attributes = getattr(value, "__dict__", None)
    if isinstance(attributes, Mapping):
        return attributes
    return None


def _technology_from_text(text: str) -> str:
    """검색 질문과 본문에서 평가 기술을 식별한다."""
    normalized = text.casefold()
    has_turboquant = "turboquant" in normalized
    has_cxl = "cxl-based" in normalized or "cxl based" in normalized
    has_cxl = has_cxl or bool(re.search(r"\bcxl\b", normalized))

    if has_turboquant and has_cxl:
        return "both"
    if has_turboquant:
        return "TurboQuant"
    if has_cxl:
        return "CXL-based"
    return "general"


def _category_from_text(text: str) -> str:
    """시장 평가 항목을 검색 질문과 결과 본문에서 식별한다."""
    normalized = text.casefold()
    for category, keywords in MARKET_CATEGORIES.items():
        if any(keyword.casefold() in normalized for keyword in keywords):
            return category
    return "general_market"


def _first_sentence(text: str) -> str:
    """근거 본문에서 EvidenceCard 주장으로 사용할 짧은 문장을 만든다."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+|\n+", compact, maxsplit=1)[0]
    return sentence[:500]


def _claim_type(value: Any, text: str) -> str:
    """검색 결과의 사실·추론 유형을 결정한다."""
    if str(value).lower() in VALID_CLAIM_TYPES:
        return str(value).lower()

    inference_words = (
        "may",
        "could",
        "potential",
        "likely",
        "expected",
        "가능",
        "잠재",
        "전망",
        "예상",
    )
    normalized = text.casefold()
    return "inference" if any(word in normalized for word in inference_words) else "fact"


def _source_type(value: Any, title: str, url: str) -> str:
    """출처 메타데이터와 URL을 EvidenceCard의 출처 유형으로 변환한다."""
    candidate = str(value or "").lower()
    if candidate in VALID_SOURCE_TYPES:
        return candidate

    combined = f"{title} {url}".lower()
    if any(token in combined for token in ("arxiv", ".edu", "doi.org")):
        return "paper"
    if any(token in combined for token in ("reuters", "techcrunch", "news")):
        return "news"
    if any(token in combined for token in ("report", "gartner", "statista")):
        return "report"
    if "blog" in combined or "medium.com" in combined:
        return "blog"
    if any(
        token in combined
        for token in (".gov", "aws.amazon", "cloud.google", "microsoft.com", "nvidia.com")
    ):
        return "official"
    return "official" if urlparse(url).netloc else "blog"


def _confidence(value: Any, content: str) -> float:
    """검색 점수 또는 본문 길이를 근거 신뢰도로 변환한다."""
    if isinstance(value, (int, float)):
        score = float(value)
        if score > 1:
            score = score / 100
        return round(max(0.0, min(score, 1.0)), 2)
    if len(content) >= 400:
        return 0.7
    if len(content) >= 120:
        return 0.6
    return 0.5


def _normalize_result(raw_result: Any, query: str) -> dict[str, Any] | None:
    """Tavily 결과 하나를 근거 카드 생성용 내부 구조로 정리한다."""
    result = _as_mapping(raw_result)
    if result is None:
        return None

    title = str(result.get("title") or "").strip()
    url = str(result.get("url") or result.get("source_url") or "").strip()
    content = str(
        result.get("content")
        or result.get("raw_content")
        or result.get("snippet")
        or ""
    ).strip()
    if not url or not content:
        return None

    combined_text = " ".join((query, title, content))
    claim = str(result.get("claim") or _first_sentence(content)).strip()
    evidence_text = str(result.get("evidence_text") or content).strip()[:2000]
    technology = _technology_from_text(
        " ".join((str(result.get("technology") or ""), combined_text))
    )
    category = _category_from_text(
        " ".join((str(result.get("category") or ""), combined_text))
    )

    return {
        "title": title or url,
        "url": url,
        "content": content,
        "evidence_text": evidence_text,
        "claim": claim or "출처 본문에서 시장 관련 정보가 확인되었습니다.",
        "technology": technology,
        "category": category,
        "claim_type": _claim_type(result.get("claim_type"), claim or content),
        "source_type": _source_type(result.get("source_type"), title, url),
        "source_locator": str(result.get("source_locator") or "web page"),
        "published_date": str(
            result.get("published_date")
            or result.get("publishedDate")
            or ""
        ),
        "confidence": _confidence(result.get("score"), content),
        "caveat": str(
            result.get("caveat")
            or "Tavily 검색 결과의 공개 본문을 별도 검증하기 전의 근거입니다."
        ),
        "query": query,
    }


def _collect_results(queries: list[str]) -> list[dict[str, Any]]:
    """질문별 Tavily 결과를 중복 제거하여 수집한다."""
    collected: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    for query in queries:
        for raw_result in _call_tavily(query):
            normalized = _normalize_result(raw_result, query)
            if normalized is None:
                continue
            url = normalized["url"].rstrip("/")
            if url in seen_urls:
                continue
            seen_urls.add(url)
            collected.append(normalized)

    return collected


def _build_evidence_cards(
    results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """정리된 검색 결과를 EvidenceCard와 시장 finding으로 변환한다."""
    cards: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []

    for finding_number, result in enumerate(results, start=1):
        technology_slug = result["technology"].lower().replace("-", "_")
        evidence_id = f"market-{technology_slug}-{finding_number:03d}"
        card = {
            "evidence_id": evidence_id,
            "technology": result["technology"],
            "perspective": "market",
            "claim": result["claim"],
            "evidence_text": result["evidence_text"],
            "source_title": result["title"],
            "source_url": result["url"],
            "source_type": result["source_type"],
            "source_locator": result["source_locator"],
            "retrieval_method": "tavily",
            "published_date": result["published_date"],
            "claim_type": result["claim_type"],
            "confidence": result["confidence"],
            "caveat": result["caveat"],
            "verification_status": "unverified",
        }
        cards.append(card)
        findings.append(
            {
                "technology": result["technology"],
                "category": result["category"],
                "claim": result["claim"],
                "claim_type": result["claim_type"],
                "evidence_ids": [evidence_id],
            }
        )

    return cards, findings


def _merge_results(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """초기 검색과 보완 검색 결과를 URL 기준으로 합친다."""
    merged: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for result in [*existing, *incoming]:
        url = str(result.get("url", "")).rstrip("/")
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        merged.append(result)
    return merged


def _generate_market_questions(state: MarketLocalState) -> MarketLocalState:
    """ResearchPlan을 바탕으로 첫 시장 평가 질문을 생성한다."""
    global_state = state["global_state"]
    queries = _build_queries(global_state)
    return {
        "queries": queries,
        "current_queries": queries,
        "retry_count": 0,
        "max_retries": 1,
        "normalized_results": [],
        "source_records": [],
        "source_urls": [],
        "evidence_cards": [],
        "findings": [],
        "missing_categories": list(MARKET_CATEGORY_LABELS),
    }


def _search_market_sources(state: MarketLocalState) -> MarketLocalState:
    """현재 질문으로 Tavily를 검색하고 오류를 내부 State에 기록한다."""
    try:
        incoming_results = _collect_results(state.get("current_queries", []))
    except _TavilyInterfaceError:
        return {
            "terminal_status": "failed",
            "terminal_summary": "Tavily 검색 도구의 공개 인터페이스를 찾지 못했습니다.",
            "terminal_error": "tavily_search.py에 공개 검색 함수가 없습니다.",
        }
    except Exception as error:  # noqa: BLE001
        error_text = str(error)
        if "TAVILY_API_KEY" in error_text:
            return {
                "terminal_status": "failed",
                "terminal_summary": (
                    "Tavily API 키가 설정되지 않아 시장 평가를 수행하지 못했습니다."
                ),
                "terminal_error": "TAVILY_API_KEY 설정이 필요합니다.",
            }
        return {
            "terminal_status": "needs_retry",
            "terminal_summary": "Tavily 검색 중 일시적인 오류가 발생했습니다.",
            "terminal_error": f"{type(error).__name__}: Tavily 호출 실패",
        }

    return {
        "normalized_results": _merge_results(
            state.get("normalized_results", []),
            incoming_results,
        )
    }


def _collect_source_urls(state: MarketLocalState) -> MarketLocalState:
    """검색 결과에서 본문과 URL이 있는 출처만 수집한다."""
    source_records = [
        result
        for result in state.get("normalized_results", [])
        if result.get("url") and result.get("content")
    ]
    return {
        "source_records": source_records,
        "source_urls": [str(result["url"]) for result in source_records],
    }


def _extract_source_information(state: MarketLocalState) -> MarketLocalState:
    """출처의 본문·제목·발행일 정보를 근거 카드 입력으로 정리한다."""
    extracted_records = []
    for result in state.get("source_records", []):
        extracted_records.append(
            {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "content": result.get("content", ""),
                "evidence_text": result.get("evidence_text", ""),
                "published_date": result.get("published_date", ""),
                "source_type": result.get("source_type", ""),
                "source_locator": result.get("source_locator", "web page"),
                "technology": result.get("technology", "general"),
                "category": result.get("category", "general_market"),
                "claim": result.get("claim", ""),
                "claim_type": result.get("claim_type", "fact"),
                "confidence": result.get("confidence", 0.0),
                "caveat": result.get("caveat", ""),
                "query": result.get("query", ""),
            }
        )
    return {"source_records": extracted_records}


def _write_market_evidence_cards(state: MarketLocalState) -> MarketLocalState:
    """정리된 출처 정보를 EvidenceCard와 finding으로 변환한다."""
    cards, findings = _build_evidence_cards(state.get("source_records", []))
    return {"evidence_cards": cards, "findings": findings}


def _check_market_sufficiency(state: MarketLocalState) -> MarketLocalState:
    """두 기술과 시장 평가 항목의 근거가 충분한지 판정한다."""
    findings = state.get("findings", [])
    technologies = {
        finding.get("technology")
        for finding in findings
        if finding.get("technology") in {"TurboQuant", "CXL-based"}
    }
    covered_categories = {
        finding.get("category")
        for finding in findings
        if finding.get("category") in MARKET_CATEGORY_LABELS
    }
    missing_categories = [
        category
        for category in MARKET_CATEGORY_LABELS
        if category not in covered_categories
    ]
    sufficient = {
        "TurboQuant",
        "CXL-based",
    }.issubset(technologies) and not missing_categories
    return {
        "missing_categories": missing_categories,
        "sufficient": sufficient,
    }


def _supplement_market_queries(state: MarketLocalState) -> MarketLocalState:
    """부족한 시장 항목을 보완하기 위한 재검색 질문을 만든다."""
    missing_categories = state.get("missing_categories", [])
    if not missing_categories:
        missing_categories = ["cloud_adoption"]

    supplement_queries = [
        CATEGORY_QUERY_TEMPLATES[category].format(
            technology="TurboQuant CXL-based"
        )
        for category in missing_categories[:MAX_TAVILY_QUERIES]
        if category in CATEGORY_QUERY_TEMPLATES
    ]
    previous_queries = state.get("queries", [])
    new_queries = [
        query for query in supplement_queries if query not in previous_queries
    ]
    return {
        "queries": [*previous_queries, *new_queries],
        "current_queries": new_queries,
        "retry_count": state.get("retry_count", 0) + 1,
    }


def _mark_insufficient_market_evidence(
    state: MarketLocalState,
) -> MarketLocalState:
    """재시도 한도에 도달한 시장 평가를 자료 부족으로 표시한다."""
    return {
        "terminal_status": "insufficient_evidence",
        "terminal_summary": "시장 평가에 필요한 공개 근거가 충분하지 않습니다.",
    }


def _route_after_sufficiency(state: MarketLocalState) -> str:
    """충분성 결과에 따라 내부 그래프의 다음 노드를 선택한다."""
    if state.get("terminal_status"):
        return "terminal"
    if state.get("sufficient"):
        return "sufficient"
    if state.get("retry_count", 0) < state.get("max_retries", 1):
        return "retry"
    return "insufficient"


def _finalize_market_result(state: MarketLocalState) -> MarketLocalState:
    """내부 그래프 결과를 외부 AgentResult 형식으로 변환한다."""
    cards = state.get("evidence_cards", [])
    findings = state.get("findings", [])
    missing_categories = state.get("missing_categories", [])
    terminal_status = state.get("terminal_status")
    terminal_summary = state.get("terminal_summary", "")
    terminal_error = state.get("terminal_error", "")

    if terminal_status:
        status = terminal_status
        summary = terminal_summary
        errors = [terminal_error] if terminal_error else []
    elif state.get("sufficient"):
        status = "ok"
        summary = f"시장 평가 근거 {len(cards)}건을 수집했습니다."
        errors = []
    else:
        status = "insufficient_evidence"
        summary = "시장 평가에 필요한 공개 근거가 충분하지 않습니다."
        errors = []

    limitations = ["공개 웹 자료와 Tavily 검색 결과만 사용했습니다."]
    if missing_categories:
        limitations.append("일부 시장 평가 항목은 직접 근거가 부족합니다.")

    sources = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "published_date": result.get("published_date", ""),
            "query": result.get("query", ""),
        }
        for result in state.get("normalized_results", [])
    ]
    result = _empty_result(
        status=status,
        summary=summary,
        queries=state.get("queries", []),
        limitations=limitations,
        errors=errors,
        gaps=[
            MARKET_CATEGORY_LABELS[category]
            for category in missing_categories
            if category in MARKET_CATEGORY_LABELS
        ],
        cards=cards,
        findings=findings,
    )
    result["market_result"]["payload"]["sources"] = sources
    return {
        "market_result": result["market_result"],
        "evidence_cards": result["evidence_cards"],
    }


def _build_market_graph():
    """시장 평가 Agent 내부의 LangGraph를 생성한다."""
    graph = StateGraph(MarketLocalState)
    graph.add_node("generate_questions", _generate_market_questions)
    graph.add_node("search", _search_market_sources)
    graph.add_node("collect_urls", _collect_source_urls)
    graph.add_node("extract_sources", _extract_source_information)
    graph.add_node("build_cards", _write_market_evidence_cards)
    graph.add_node("check_sufficiency", _check_market_sufficiency)
    graph.add_node("supplement_queries", _supplement_market_queries)
    graph.add_node("mark_insufficient", _mark_insufficient_market_evidence)
    graph.add_node("finalize", _finalize_market_result)

    graph.add_edge(START, "generate_questions")
    graph.add_edge("generate_questions", "search")
    graph.add_edge("search", "collect_urls")
    graph.add_edge("collect_urls", "extract_sources")
    graph.add_edge("extract_sources", "build_cards")
    graph.add_edge("build_cards", "check_sufficiency")
    graph.add_conditional_edges(
        "check_sufficiency",
        _route_after_sufficiency,
        {
            "sufficient": "finalize",
            "retry": "supplement_queries",
            "insufficient": "mark_insufficient",
            "terminal": "finalize",
        },
    )
    graph.add_edge("supplement_queries", "search")
    graph.add_edge("mark_insufficient", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def _empty_result(
    *,
    status: str,
    summary: str,
    queries: list[str],
    limitations: list[str] | None = None,
    errors: list[str] | None = None,
    gaps: list[str] | None = None,
    cards: list[dict[str, Any]] | None = None,
    findings: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """AgentResult의 모든 공통 필드를 채운 State 업데이트를 만든다."""
    result_cards = cards or []
    return {
        "market_result": {
            "agent_name": "market_evaluation",
            "status": status,
            "summary": summary,
            "evidence_ids": [card["evidence_id"] for card in result_cards],
            "limitations": limitations or [],
            "errors": errors or [],
            "payload": {
                "queries": queries,
                "findings": findings or [],
                "gaps": gaps or [],
            },
        },
        "evidence_cards": result_cards,
    }


def market_evaluation_agent(state: GlobalState) -> dict[str, Any]:
    """내부 시장 평가 그래프를 실행하고 AgentResult를 반환한다."""
    technical_result = state.get("technical_result")
    queries = _build_queries(state)

    if technical_result is None:
        return _empty_result(
            status="failed",
            summary="기술 조사 결과가 없어 시장 평가를 시작하지 못했습니다.",
            queries=queries,
            errors=["technical_result 입력이 없습니다."],
        )

    technical_status = technical_result.get("status")
    technical_payload = technical_result.get("payload") or {}
    if (
        technical_status in {"failed", "insufficient_evidence"}
        and not technical_result.get("evidence_ids")
        and not technical_payload.get("technologies")
    ):
        return _empty_result(
            status="insufficient_evidence",
            summary="기술 조사 근거가 부족하여 시장 평가를 수행할 수 없습니다.",
            queries=queries,
            limitations=["기술 조사 Agent가 사용할 수 있는 근거를 반환하지 않았습니다."],
            gaps=["TurboQuant 기술 근거", "CXL-based 기술 근거"],
        )

    local_graph = _build_market_graph()
    local_state: MarketLocalState = {
        "global_state": state,
        "queries": queries,
        "current_queries": queries,
        "retry_count": 0,
        "max_retries": 1,
    }
    graph_result = local_graph.invoke(local_state)
    return {
        "market_result": graph_result["market_result"],
        "evidence_cards": graph_result.get("evidence_cards", []),
    }
