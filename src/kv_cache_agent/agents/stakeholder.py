"""Tavily 검색 결과를 이해관계자 평가 근거로 변환하는 에이전트."""

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
MAX_RETRIES = 1

STAKEHOLDERS = [
    "클라우드 사업자",
    "AI 모델 개발사",
    "하드웨어 제조사",
    "클라우드 고객",
    "오픈소스 개발자",
    "연구자·투자자",
]

DEFAULT_STAKEHOLDER_QUERIES = [
    "cloud providers cloud customers TurboQuant CXL-based KV cache adoption concerns",
    "AI model developers open source developers TurboQuant CXL-based integration opinions",
    "hardware manufacturers researchers investors CXL-based TurboQuant market outlook",
]

STAKEHOLDER_QUERY_TEMPLATES = {
    "클라우드 사업자": (
        "cloud providers TurboQuant CXL-based KV cache adoption "
        "cost and operational concerns"
    ),
    "AI 모델 개발사": (
        "AI model developers TurboQuant CXL-based KV cache "
        "integration and model quality concerns"
    ),
    "하드웨어 제조사": (
        "hardware manufacturers CXL-based memory TurboQuant "
        "product and ecosystem response"
    ),
    "클라우드 고객": (
        "cloud customers TurboQuant CXL-based inference cost "
        "performance and service quality"
    ),
    "오픈소스 개발자": (
        "open source developers TurboQuant CXL-based framework "
        "support and adoption"
    ),
    "연구자·투자자": (
        "researchers investors TurboQuant CXL-based KV cache "
        "technology outlook and risks"
    ),
}

STAKEHOLDER_KEYWORDS = {
    "클라우드 사업자": (
        "cloud provider",
        "cloud providers",
        "hyperscaler",
        "aws",
        "google cloud",
        "microsoft azure",
        "클라우드 사업자",
    ),
    "AI 모델 개발사": (
        "model developer",
        "model developers",
        "foundation model",
        "llm developer",
        "ai model",
        "ai 모델 개발사",
    ),
    "하드웨어 제조사": (
        "hardware manufacturer",
        "hardware vendor",
        "semiconductor",
        "memory vendor",
        "하드웨어 제조사",
    ),
    "클라우드 고객": (
        "cloud customer",
        "cloud customers",
        "enterprise customer",
        "cloud user",
        "클라우드 고객",
    ),
    "오픈소스 개발자": (
        "open source developer",
        "open-source developer",
        "open source community",
        "opensource",
        "오픈소스 개발자",
    ),
    "연구자·투자자": (
        "researcher",
        "researchers",
        "investor",
        "investors",
        "analyst",
        "연구자",
        "투자자",
    ),
}

VALID_CLAIM_TYPES = {"fact", "inference"}
VALID_SOURCE_TYPES = {"paper", "official", "news", "blog", "report"}
TAVILY_FUNCTION_NAMES = ("search_web", "search_tavily", "tavily_search", "search")


class StakeholderLocalState(TypedDict, total=False):
    """이해관계자 평가 Agent 내부 그래프에서만 사용하는 임시 State."""

    global_state: GlobalState
    stakeholders: list[str]
    queries: list[str]
    current_queries: list[str]
    normalized_results: list[dict[str, Any]]
    source_records: list[dict[str, Any]]
    findings: list[dict[str, Any]]
    evidence_cards: list[dict[str, Any]]
    missing_stakeholders: list[str]
    retry_count: int
    max_retries: int
    sufficient: bool
    terminal_status: str
    terminal_summary: str
    terminal_error: str
    stakeholder_result: dict[str, Any]


class _TavilyInterfaceError(RuntimeError):
    """Tavily 도구에서 호출 가능한 공개 함수를 찾지 못했을 때 발생한다."""


def _build_queries(state: GlobalState) -> list[str]:
    """ResearchPlan의 질문을 사용하고 부족하면 기본 질문을 보완한다."""
    research_plan = state.get("research_plan", {})
    search_questions = research_plan.get("search_questions", {})
    planned_queries = search_questions.get("stakeholder", [])

    queries: list[str] = []
    for query in planned_queries:
        normalized_query = str(query).strip()
        if normalized_query and normalized_query not in queries:
            queries.append(normalized_query)

    for query in DEFAULT_STAKEHOLDER_QUERIES:
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


def _first_sentence(text: str) -> str:
    """본문에서 근거 카드 주장으로 사용할 짧은 문장을 만든다."""
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return ""
    sentence = re.split(r"(?<=[.!?])\s+|\n+", compact, maxsplit=1)[0]
    return sentence[:500]


def _claim_type(value: Any, text: str) -> str:
    """검색 결과를 사실과 추론 중 하나로 구분한다."""
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
    """출처 메타데이터와 URL을 EvidenceCard 유형으로 변환한다."""
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


def _stakeholders_from_text(text: str) -> list[str]:
    """검색 결과에서 관련 이해관계자 유형을 찾는다."""
    normalized = text.casefold()
    return [
        stakeholder
        for stakeholder, keywords in STAKEHOLDER_KEYWORDS.items()
        if any(keyword.casefold() in normalized for keyword in keywords)
    ]


def _position_from_text(text: str) -> str:
    """본문의 표현을 기대 효과·우려·중립 입장으로 분류한다."""
    normalized = text.casefold()
    concern_words = (
        "concern",
        "risk",
        "barrier",
        "challenge",
        "limitation",
        "cost",
        "우려",
        "위험",
        "장벽",
        "한계",
    )
    benefit_words = (
        "benefit",
        "advantage",
        "saving",
        "adoption",
        "효과",
        "장점",
        "절감",
        "채택",
    )
    has_concern = any(word in normalized for word in concern_words)
    has_benefit = any(word in normalized for word in benefit_words)
    if has_concern and has_benefit:
        return "mixed"
    if has_concern:
        return "concern"
    if has_benefit:
        return "benefit"
    return "uncertain"


def _normalize_result(raw_result: Any, query: str) -> dict[str, Any] | None:
    """Tavily 검색 결과 하나를 이해관계자 근거용 구조로 정리한다."""
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

    raw_stakeholders = result.get("stakeholders")
    if isinstance(raw_stakeholders, Sequence) and not isinstance(
        raw_stakeholders, (str, bytes)
    ):
        stakeholders = [
            str(item)
            for item in raw_stakeholders
            if str(item) in STAKEHOLDERS
        ]
    else:
        stakeholders = []
    if not stakeholders:
        # 검색어에는 여러 이해관계자 유형이 함께 들어갈 수 있으므로,
        # 검색어만으로 유형을 부여하면 근거 본문과 무관한 분류가 생긴다.
        stakeholders = _stakeholders_from_text(" ".join((title, content)))

    if not stakeholders:
        return None

    claim = str(result.get("claim") or _first_sentence(content)).strip()
    evidence_text = str(result.get("evidence_text") or content).strip()[:2000]
    combined_text = " ".join((query, title, content))
    return {
        "title": title or url,
        "url": url,
        "content": content,
        "evidence_text": evidence_text,
        "claim": claim or "이해관계자 관련 의견이 공개 자료에서 확인되었습니다.",
        "technology": _technology_from_text(
            " ".join((str(result.get("technology") or ""), combined_text))
        ),
        "stakeholders": stakeholders,
        "position": _position_from_text(combined_text),
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
            or "공개 자료를 별도 검증하기 전의 이해관계자 근거입니다."
        ),
        "query": query,
    }


def _collect_results(queries: list[str]) -> list[dict[str, Any]]:
    """질문별 Tavily 결과를 수집하고 URL 중복을 제거한다."""
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


def _generate_stakeholder_list(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """평가할 6개 이해관계자와 첫 검색 질문을 생성한다."""
    global_state = state["global_state"]
    queries = _build_queries(global_state)
    return {
        "stakeholders": list(STAKEHOLDERS),
        "queries": queries,
        "current_queries": queries,
        "normalized_results": [],
        "source_records": [],
        "findings": [],
        "evidence_cards": [],
        "missing_stakeholders": list(STAKEHOLDERS),
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    }


def _search_stakeholder_sources(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """현재 질문으로 Tavily를 검색하고 API 오류를 기록한다."""
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
                    "Tavily API 키가 설정되지 않아 이해관계자 평가를 수행하지 못했습니다."
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


def _extract_stakeholder_sources(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """기업 발표·뉴스·업계 의견에 필요한 출처 정보를 추출한다."""
    records = [
        result
        for result in state.get("normalized_results", [])
        if result.get("url") and result.get("content") and result.get("stakeholders")
    ]
    return {"source_records": records}


def _build_stakeholder_evidence(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """출처별 기대 효과·우려·입장을 EvidenceCard와 finding으로 만든다."""
    cards: list[dict[str, Any]] = []
    findings: list[dict[str, Any]] = []
    card_number = 0

    for result in state.get("source_records", []):
        for stakeholder in result["stakeholders"]:
            card_number += 1
            technology_slug = result["technology"].lower().replace("-", "_")
            stakeholder_slug = re.sub(r"[^a-z0-9가-힣]+", "_", stakeholder.lower())
            evidence_id = (
                f"stakeholder-{stakeholder_slug}-{technology_slug}-{card_number:03d}"
            )
            claim = f"[{stakeholder}] {result['claim']}"
            card = {
                "evidence_id": evidence_id,
                "technology": result["technology"],
                "perspective": "stakeholder",
                "claim": claim,
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
                    "stakeholder": stakeholder,
                    "technology": result["technology"],
                    "expectations": [claim]
                    if result["position"] in {"benefit", "mixed"}
                    else [],
                    "concerns": [claim]
                    if result["position"] in {"concern", "mixed"}
                    else [],
                    "position": result["position"],
                    "claim_type": result["claim_type"],
                    "evidence_ids": [evidence_id],
                }
            )

    return {"evidence_cards": cards, "findings": findings}


def _check_stakeholder_evidence(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """이해관계자별 근거가 모두 확보되었는지 확인한다."""
    covered = {
        finding.get("stakeholder")
        for finding in state.get("findings", [])
        if finding.get("stakeholder") in STAKEHOLDERS
    }
    missing = [stakeholder for stakeholder in STAKEHOLDERS if stakeholder not in covered]
    return {
        "missing_stakeholders": missing,
        "sufficient": not missing,
    }


def _supplement_stakeholder_queries(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """근거가 부족한 이해관계자에 대한 보완 검색 질문을 만든다."""
    missing = state.get("missing_stakeholders", [])
    if not missing:
        missing = ["연구자·투자자"]
    supplement_queries = [
        STAKEHOLDER_QUERY_TEMPLATES[stakeholder]
        for stakeholder in missing[:MAX_TAVILY_QUERIES]
        if stakeholder in STAKEHOLDER_QUERY_TEMPLATES
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


def _mark_uncertain_stakeholders(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """재검색 후에도 근거가 없는 이해관계자를 불확실성으로 표시한다."""
    findings = list(state.get("findings", []))
    existing = {finding.get("stakeholder") for finding in findings}
    for stakeholder in state.get("missing_stakeholders", []):
        if stakeholder in existing:
            continue
        findings.append(
            {
                "stakeholder": stakeholder,
                "technology": "general",
                "expectations": [],
                "concerns": [],
                "position": "uncertain",
                "claim_type": "inference",
                "evidence_ids": [],
                "uncertainty": "직접적인 공개 발언이나 충분한 근거를 확인하지 못함",
            }
        )
    return {
        "findings": findings,
        "terminal_status": "insufficient_evidence",
        "terminal_summary": "일부 이해관계자의 공개 근거가 충분하지 않습니다.",
    }


def _route_after_stakeholder_check(state: StakeholderLocalState) -> str:
    """충분성 결과에 따라 다음 내부 그래프 노드를 선택한다."""
    if state.get("terminal_status"):
        return "terminal"
    if state.get("sufficient"):
        return "sufficient"
    if state.get("retry_count", 0) < state.get("max_retries", MAX_RETRIES):
        return "retry"
    return "uncertain"


def _finalize_stakeholder_result(
    state: StakeholderLocalState,
) -> StakeholderLocalState:
    """내부 그래프 결과를 외부 AgentResult 형식으로 변환한다."""
    findings = state.get("findings", [])
    cards = state.get("evidence_cards", [])
    missing = state.get("missing_stakeholders", [])
    terminal_status = state.get("terminal_status")
    terminal_summary = state.get("terminal_summary", "")
    terminal_error = state.get("terminal_error", "")

    if terminal_status:
        status = terminal_status
        summary = terminal_summary
        errors = [terminal_error] if terminal_error else []
    elif state.get("sufficient"):
        status = "ok"
        summary = f"이해관계자 평가 근거 {len(cards)}건을 수집했습니다."
        errors = []
    else:
        status = "insufficient_evidence"
        summary = "이해관계자별 공개 근거가 충분하지 않습니다."
        errors = []

    limitations = ["공개 기업 발표·뉴스·업계 자료와 Tavily 검색 결과만 사용했습니다."]
    if missing:
        limitations.append("일부 이해관계자는 직접적인 공개 발언을 확인하지 못했습니다.")

    sources = [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "published_date": result.get("published_date", ""),
            "query": result.get("query", ""),
        }
        for result in state.get("normalized_results", [])
    ]
    stakeholder_result = {
        "agent_name": "stakeholder_evaluation",
        "status": status,
        "summary": summary,
        "evidence_ids": [card["evidence_id"] for card in cards],
        "limitations": limitations,
        "errors": errors,
        "payload": {
            "stakeholders": state.get("stakeholders", list(STAKEHOLDERS)),
            "queries": state.get("queries", []),
            "findings": findings,
            "gaps": missing,
            "sources": sources,
        },
    }
    return {
        "stakeholder_result": stakeholder_result,
        "evidence_cards": cards,
    }


def _build_stakeholder_graph():
    """이해관계자 평가 Agent 내부의 LangGraph를 생성한다."""
    graph = StateGraph(StakeholderLocalState)
    graph.add_node("generate_stakeholders", _generate_stakeholder_list)
    graph.add_node("search", _search_stakeholder_sources)
    graph.add_node("extract_sources", _extract_stakeholder_sources)
    graph.add_node("analyze_positions", _build_stakeholder_evidence)
    graph.add_node("check_evidence", _check_stakeholder_evidence)
    graph.add_node("supplement_queries", _supplement_stakeholder_queries)
    graph.add_node("mark_uncertain", _mark_uncertain_stakeholders)
    graph.add_node("finalize", _finalize_stakeholder_result)

    graph.add_edge(START, "generate_stakeholders")
    graph.add_edge("generate_stakeholders", "search")
    graph.add_edge("search", "extract_sources")
    graph.add_edge("extract_sources", "analyze_positions")
    graph.add_edge("analyze_positions", "check_evidence")
    graph.add_conditional_edges(
        "check_evidence",
        _route_after_stakeholder_check,
        {
            "sufficient": "finalize",
            "retry": "supplement_queries",
            "uncertain": "mark_uncertain",
            "terminal": "finalize",
        },
    )
    graph.add_edge("supplement_queries", "search")
    graph.add_edge("mark_uncertain", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


def stakeholder_evaluation_agent(state: GlobalState) -> dict[str, Any]:
    """내부 이해관계자 평가 그래프를 실행하고 AgentResult를 반환한다."""
    technical_result = state.get("technical_result")
    if technical_result is None:
        return {
            "stakeholder_result": {
                "agent_name": "stakeholder_evaluation",
                "status": "failed",
                "summary": "기술 조사 결과가 없어 이해관계자 평가를 시작하지 못했습니다.",
                "evidence_ids": [],
                "limitations": [],
                "errors": ["technical_result 입력이 없습니다."],
                "payload": {
                    "stakeholders": list(STAKEHOLDERS),
                    "queries": [],
                    "findings": [],
                    "gaps": list(STAKEHOLDERS),
                    "sources": [],
                },
            },
            "evidence_cards": [],
        }

    technical_payload = technical_result.get("payload") or {}
    if (
        technical_result.get("status") in {"failed", "insufficient_evidence"}
        and not technical_result.get("evidence_ids")
        and not technical_payload.get("technologies")
    ):
        return {
            "stakeholder_result": {
                "agent_name": "stakeholder_evaluation",
                "status": "insufficient_evidence",
                "summary": "기술 조사 근거가 부족하여 이해관계자 평가를 수행할 수 없습니다.",
                "evidence_ids": [],
                "limitations": [
                    "기술 조사 Agent가 사용할 수 있는 근거를 반환하지 않았습니다."
                ],
                "errors": [],
                "payload": {
                    "stakeholders": list(STAKEHOLDERS),
                    "queries": [],
                    "findings": [],
                    "gaps": list(STAKEHOLDERS),
                    "sources": [],
                },
            },
            "evidence_cards": [],
        }

    graph = _build_stakeholder_graph()
    result = graph.invoke({"global_state": state})
    return {
        "stakeholder_result": result["stakeholder_result"],
        "evidence_cards": result.get("evidence_cards", []),
    }
