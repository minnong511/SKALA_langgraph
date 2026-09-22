"""TurboQuant와 CXL-based KV Cache의 클라우드 적용성을 평가하는 에이전트."""

from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm
from kv_cache_agent.schemas.outputs import EvidenceCard
from kv_cache_agent.schemas.tool_outputs import WebSearchResult
from kv_cache_agent.tools.paper_retriever import retrieve_paper_chunks
from kv_cache_agent.tools.tavily_search import TavilySearchError, search_web

# 프롬프트 문구를 Python 코드와 분리하여 역할과 제약 조건을 쉽게 검토할 수 있게 한다.
PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "cloud_domain.yaml"

# 개발 문서의 검색 제한에 따라 최초 질문 2개와 부족 시 재검색 질문 1개,
# 총 3개까지만 Tavily에 전달한다.
MAX_SEARCH_QUERIES = 3
MAX_INITIAL_SEARCH_QUERIES = 2
MAX_RETRIES = 1
MAX_RESULTS_PER_QUERY = 5
VECTOR_TOP_K = 3

# 도식의 "클라우드 사용 시나리오 생성" 단계에서 사용하는 시나리오 정의다.
# 두 기술을 반드시 같은 시나리오와 질문으로 평가할 수 있도록 코드에서 고정한다.
CLOUD_SCENARIOS = [
    {
        "id": "cloud_llm_serving",
        "label": "클라우드 LLM 추론 서비스 적합성",
        "search_focus": "cloud LLM inference serving suitability",
    },
    {
        "id": "gpu_memory_cost",
        "label": "GPU 메모리 비용 절감 가능성",
        "search_focus": "GPU memory capacity and serving cost",
    },
    {
        "id": "long_context",
        "label": "긴 컨텍스트 처리 적합성",
        "search_focus": "long context KV cache capacity",
    },
    {
        "id": "multi_tenancy",
        "label": "멀티테넌시 환경 적합성",
        "search_focus": "multi tenancy concurrent requests",
    },
    {
        "id": "latency_throughput_slo",
        "label": "지연시간·처리량·SLO 영향",
        "search_focus": "latency throughput service level objective",
    },
    {
        "id": "operational_complexity",
        "label": "클라우드 사업자 운영 난이도",
        "search_focus": "deployment operational complexity hardware requirements",
    },
    {
        "id": "customer_impact",
        "label": "고객 체감 효과",
        "search_focus": "customer performance price and context length impact",
    },
]

CLOUD_EVALUATION_CRITERIA = [scenario["id"] for scenario in CLOUD_SCENARIOS]

# Supervisor가 별도의 질문을 만들지 않았을 때 사용할 기본 검색어다.
# 검색어 하나에 한 기술만 넣지 않고, 마지막 검색어에서 두 기술을 같은 기준으로
# 비교할 수 있는 클라우드 운영 자료를 함께 찾도록 구성했다.
DEFAULT_CLOUD_QUERIES = [
    (
        "TurboQuant KV cache quantization cloud LLM inference "
        "GPU memory latency throughput"
    ),
    (
        "CXL based KV cache cloud LLM inference long context "
        "latency throughput"
    ),
]

Technology = Literal["TurboQuant", "CXL-based", "both", "general"]
ClaimType = Literal["fact", "inference"]
EvidenceSourceType = Literal["paper", "official", "news", "blog", "report"]
CloudCriterion = Literal[
    "cloud_llm_serving",
    "gpu_memory_cost",
    "long_context",
    "multi_tenancy",
    "latency_throughput_slo",
    "operational_complexity",
    "customer_impact",
]


class CloudDomainFinding(BaseModel):
    """LLM이 반환하는 클라우드 평가 주장 한 건의 구조."""

    technology: Technology
    criterion: CloudCriterion
    claim: str
    evidence_text: str
    source_url: str
    source_locator: str = ""
    source_type: EvidenceSourceType
    claim_type: ClaimType
    confidence: float = Field(ge=0.0, le=1.0)
    caveat: str = ""


class CloudDomainExtraction(BaseModel):
    """LLM의 전체 클라우드 평가 결과를 검증 가능한 형태로 제한한다."""

    summary: str
    comparison: dict[str, Any] = Field(default_factory=dict)
    findings: list[CloudDomainFinding] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


class CloudDomainGraphState(TypedDict, total=False):
    """클라우드 도메인 전용 LangGraph 노드 사이에서만 사용하는 LocalState."""

    global_state: GlobalState
    technical_cards: list[EvidenceCard]
    scenarios: list[dict[str, str]]
    initial_queries: list[str]
    active_queries: list[str]
    retry_queries: list[str]
    retry_count: int
    retry_possible: bool
    web_results: list[WebSearchResult]
    vector_chunks: list[dict[str, Any]]
    related_source_count: int
    evidence_cards: list[EvidenceCard]
    findings: list[CloudDomainFinding]
    comparison: dict[str, Any]
    summary: str
    missing_coverage: list[tuple[str, str]]
    limitations: list[str]
    errors: list[str]
    skipped_findings: list[str]
    fatal_error: str
    cloud_domain_result: dict[str, Any]


def _load_system_prompt() -> str:
    """YAML에서 클라우드 평가용 시스템 프롬프트를 읽는다."""
    with PROMPT_PATH.open(encoding="utf-8") as prompt_file:
        prompt_config = yaml.safe_load(prompt_file) or {}
    return str(prompt_config.get("system_prompt", ""))


def _build_cloud_scenarios(state: GlobalState) -> list[dict[str, str]]:
    """사용자 요청에 적용할 7개 클라우드 사용 시나리오를 생성한다.

    평가 기준 자체는 팀 문서에서 확정되었으므로 임의로 빼거나 추가하지 않는다.
    사용자 질문은 이후 검색 질문과 LLM 평가 문맥에 반영한다.
    """
    del state  # 현재는 고정된 클라우드 시나리오를 모든 요청에 동일하게 적용한다.
    return [dict(scenario) for scenario in CLOUD_SCENARIOS]


def _build_queries(state: GlobalState) -> list[str]:
    """시나리오를 묶은 최초 검색 질문을 최대 2개 생성한다."""
    research_plan = state.get("research_plan", {})
    search_questions = research_plan.get("search_questions", {})
    planned_queries = search_questions.get("cloud_domain", [])

    # 공백 질문과 같은 질문을 제거하여 불필요한 Tavily 호출을 막는다.
    unique_queries: list[str] = []
    for query in planned_queries:
        clean_query = str(query).strip()
        if clean_query and clean_query not in unique_queries:
            unique_queries.append(clean_query)

    # 세 번째 검색 질문은 부족한 시나리오가 발견됐을 때 재검색용으로 남겨 둔다.
    return (unique_queries or DEFAULT_CLOUD_QUERIES)[:MAX_INITIAL_SEARCH_QUERIES]


def _build_retry_query(missing_coverage: list[tuple[str, str]]) -> str:
    """근거가 부족한 기술·시나리오만 대상으로 한 재검색 질문을 만든다."""
    scenario_by_id = {scenario["id"]: scenario for scenario in CLOUD_SCENARIOS}
    technologies = sorted({technology for technology, _ in missing_coverage})
    search_focuses = list(
        dict.fromkeys(
            scenario_by_id[criterion]["search_focus"]
            for _, criterion in missing_coverage
            if criterion in scenario_by_id
        )
    )
    return (
        f"{' and '.join(technologies)} KV cache cloud evidence "
        f"{' '.join(search_focuses)}"
    ).strip()


def _technical_cards_from_state(state: GlobalState) -> list[EvidenceCard]:
    """전체 근거 중 기술 조사 에이전트가 만든 근거만 선택한다."""
    return [
        card
        for card in state.get("evidence_cards", [])
        if card.get("perspective") == "technical"
        and card.get("technology") in {"TurboQuant", "CXL-based", "both"}
    ]


def _search_cloud_sources(
    queries: list[str],
) -> tuple[list[WebSearchResult], list[str]]:
    """공용 Tavily 도구로 검색하고 URL이 같은 결과는 한 번만 남긴다."""
    results: list[WebSearchResult] = []
    errors: list[str] = []
    seen_urls: set[str] = set()

    for query in queries:
        try:
            query_results = search_web(
                query,
                max_results=MAX_RESULTS_PER_QUERY,
                search_depth="advanced",
                topic="general",
                include_raw_content=False,
            )
        except (TavilySearchError, ValueError) as error:
            # 한 검색어가 실패해도 나머지 검색어는 계속 실행하고 오류를 결과에 남긴다.
            errors.append(f"검색어 '{query}' 처리 실패: {error}")
            continue

        for result in query_results:
            url = str(result.get("url", "")).strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(result)

    return results, errors


def _search_vector_sources(
    queries: list[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """같은 검색 질문으로 논문 Vector DB를 조회한다.

    Vector DB가 아직 생성되지 않았거나 조회 중 오류가 나더라도 Tavily 결과로
    평가를 계속할 수 있도록, 예외를 밖으로 던지지 않고 오류 목록으로 반환한다.
    """
    try:
        chunks = retrieve_paper_chunks(queries, top_k=VECTOR_TOP_K)
    except Exception as error:  # noqa: BLE001
        return [], [f"Vector DB 조회 실패: {error}"]
    return chunks, []


def _merge_web_results(
    current: list[WebSearchResult],
    added: list[WebSearchResult],
) -> list[WebSearchResult]:
    """최초 검색과 재검색 결과를 URL 기준으로 합친다."""
    merged: list[WebSearchResult] = []
    seen_urls: set[str] = set()
    for result in [*current, *added]:
        url = str(result.get("url", "")).strip()
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        merged.append(result)
    return merged


def _merge_vector_chunks(
    current: list[dict[str, Any]],
    added: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """최초 조회와 재조회의 논문 chunk를 chunk_id 기준으로 합친다."""
    merged: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()
    for chunk in [*current, *added]:
        chunk_id = str(chunk.get("chunk_id", "")).strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        merged.append(chunk)
    return merged


def _build_source_context(
    technical_cards: list[EvidenceCard],
    web_results: list[WebSearchResult],
    vector_chunks: list[dict[str, Any]],
) -> str:
    """LLM이 허용된 출처 URL만 인용하도록 입력 근거를 명시적으로 표시한다."""
    context_parts: list[str] = []

    for card in technical_cards:
        context_parts.append(
            "\n".join(
                [
                    "[technical_evidence]",
                    f"technology={card.get('technology', '')}",
                    f"source_title={card.get('source_title', '')}",
                    f"source_url={card.get('source_url', '')}",
                    f"source_locator={card.get('source_locator', '')}",
                    f"claim={card.get('claim', '')}",
                    f"evidence_text={card.get('evidence_text', '')}",
                    f"caveat={card.get('caveat', '')}",
                ]
            )
        )

    for result in web_results:
        context_parts.append(
            "\n".join(
                [
                    "[web_evidence]",
                    f"source_title={result.get('title', '')}",
                    f"source_url={result.get('url', '')}",
                    f"published_date={result.get('published_date', '')}",
                    f"search_score={result.get('score', 0.0)}",
                    f"content={result.get('content', '')}",
                ]
            )
        )

    for chunk in vector_chunks:
        metadata = chunk.get("metadata", {})
        source_url = str(
            metadata.get("source_url") or metadata.get("source_path", "")
        )
        context_parts.append(
            "\n".join(
                [
                    "[vector_evidence]",
                    f"source_chunk_id={chunk.get('chunk_id', '')}",
                    f"source_title={metadata.get('source_title', '')}",
                    f"source_url={source_url}",
                    f"source_locator={metadata.get('source_locator', '')}",
                    f"content={chunk.get('content', '')}",
                ]
            )
        )

    return "\n\n---\n\n".join(context_parts)


def _extract_cloud_assessment(
    state: GlobalState,
    technical_cards: list[EvidenceCard],
    web_results: list[WebSearchResult],
    vector_chunks: list[dict[str, Any]],
    required_coverage: list[tuple[str, str]] | None = None,
) -> CloudDomainExtraction:
    """기술 근거와 웹 근거를 구조화된 클라우드 평가로 변환한다."""
    llm = get_llm().with_structured_output(CloudDomainExtraction)
    coverage_instruction = ""
    if required_coverage:
        coverage_text = ", ".join(
            f"{technology}:{criterion}"
            for technology, criterion in required_coverage
        )
        coverage_instruction = (
            "\n이번 재평가에서는 다음 기술·시나리오의 누락 근거를 우선 보완하라: "
            f"{coverage_text}\n"
        )

    prompt = (
        f"사용자 질문: {state.get('user_query', '')}\n\n"
        "아래에 제공된 기술 근거와 웹 검색 결과만 사용하여 TurboQuant와 "
        "CXL-based KV Cache의 클라우드 LLM 서빙 적합성을 평가하라.\n"
        f"반드시 사용할 평가 기준: {', '.join(CLOUD_EVALUATION_CRITERIA)}\n"
        "두 기술에 동일한 기준을 적용하고, 각 finding의 source_url에는 아래 "
        "문맥에 실제로 존재하는 URL 하나를 그대로 넣어라.\n"
        "출처가 직접 말하지 않은 클라우드 적용 해석은 claim_type을 inference로 "
        "표시하고 caveat에 추론의 한계를 작성하라.\n\n"
        f"{coverage_instruction}"
        "각 finding에는 source_url뿐 아니라 논문이면 source_locator도 입력하라.\n\n"
        "사용 가능한 근거:\n"
        f"{_build_source_context(technical_cards, web_results, vector_chunks)}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=_load_system_prompt()),
            HumanMessage(content=prompt),
        ]
    )

    if isinstance(response, CloudDomainExtraction):
        return response
    return CloudDomainExtraction.model_validate(response)


def _build_evidence_cards(
    extraction: CloudDomainExtraction,
    technical_cards: list[EvidenceCard],
    web_results: list[WebSearchResult],
    vector_chunks: list[dict[str, Any]],
) -> tuple[list[EvidenceCard], list[str]]:
    """구조화 결과를 공통 EvidenceCard 형식으로 변환한다."""
    technical_by_url = {
        str(card.get("source_url", "")).strip(): card
        for card in technical_cards
        if str(card.get("source_url", "")).strip()
    }
    web_by_url = {
        str(result.get("url", "")).strip(): result
        for result in web_results
        if str(result.get("url", "")).strip()
    }
    vector_by_url_and_locator: dict[tuple[str, str], dict[str, Any]] = {}
    vector_by_url: dict[str, dict[str, Any]] = {}
    for chunk in vector_chunks:
        metadata = chunk.get("metadata", {})
        source_url = str(
            metadata.get("source_url") or metadata.get("source_path", "")
        ).strip()
        source_locator = str(metadata.get("source_locator", "")).strip()
        if not source_url:
            continue
        vector_by_url_and_locator[(source_url, source_locator)] = chunk
        vector_by_url.setdefault(source_url, chunk)

    cards: list[EvidenceCard] = []
    skipped_findings: list[str] = []
    seen_findings: set[tuple[str, str, str]] = set()

    for finding in extraction.findings:
        source_url = finding.source_url.strip()
        requested_locator = finding.source_locator.strip()
        finding_key = (finding.technology, finding.claim, source_url)

        # 제공하지 않은 URL을 LLM이 새로 만든 경우 근거 카드에 포함하지 않는다.
        if (
            source_url not in technical_by_url
            and source_url not in web_by_url
            and source_url not in vector_by_url
        ):
            skipped_findings.append(finding.claim)
            continue
        if finding_key in seen_findings:
            continue
        seen_findings.add(finding_key)

        if source_url in technical_by_url:
            source = technical_by_url[source_url]
            source_title = str(source.get("source_title", ""))
            source_type = source.get("source_type", "paper")
            source_locator = str(source.get("source_locator", ""))
            retrieval_method = source.get("retrieval_method", "faiss")
            published_date = str(source.get("published_date", ""))
        elif source_url in vector_by_url:
            vector_source = vector_by_url_and_locator.get(
                (source_url, requested_locator),
                vector_by_url[source_url],
            )
            metadata = vector_source.get("metadata", {})
            source_title = str(metadata.get("source_title", ""))
            source_type = "paper"
            source_locator = str(metadata.get("source_locator", ""))
            retrieval_method = "faiss"
            published_date = str(metadata.get("published_date", ""))
        else:
            web_source = web_by_url[source_url]
            source_title = str(web_source.get("title", ""))
            # WebSearchResult의 source_type은 'web'이지만 EvidenceCard는 'web'을
            # 허용하지 않으므로 LLM이 내용에 맞게 분류한 공통 source_type을 사용한다.
            source_type = finding.source_type
            source_locator = "Tavily 검색 결과 요약"
            retrieval_method = "tavily"
            published_date = str(web_source.get("published_date", ""))

        technology_slug = finding.technology.lower().replace("-", "_")
        evidence_id = f"cloud-domain-{technology_slug}-{len(cards) + 1:03d}"
        cards.append(
            {
                "evidence_id": evidence_id,
                "technology": finding.technology,
                "perspective": "cloud_domain",
                "claim": finding.claim,
                "evidence_text": finding.evidence_text,
                "source_title": source_title,
                "source_url": source_url,
                "source_type": source_type,
                "source_locator": source_locator,
                "retrieval_method": retrieval_method,
                "published_date": published_date,
                "claim_type": finding.claim_type,
                "confidence": finding.confidence,
                "caveat": finding.caveat,
                "verification_status": "unverified",
            }
        )

    return cards, skipped_findings


def _missing_scenario_coverage(
    cards: list[EvidenceCard],
    scenarios: list[dict[str, str]],
    findings: list[CloudDomainFinding],
) -> list[tuple[str, str]]:
    """7개 시나리오가 두 기술 모두에 대해 평가됐는지 확인한다."""
    valid_card_keys = {
        (card.get("technology"), card.get("claim"), card.get("source_url"))
        for card in cards
    }
    covered: set[tuple[str, str]] = set()

    # 출처 검사를 통과하여 실제 EvidenceCard로 만들어진 finding만 완료로 인정한다.
    for finding in findings:
        finding_key = (finding.technology, finding.claim, finding.source_url.strip())
        if finding_key not in valid_card_keys:
            continue
        if finding.technology == "both":
            covered.add(("TurboQuant", finding.criterion))
            covered.add(("CXL-based", finding.criterion))
        elif finding.technology in {"TurboQuant", "CXL-based"}:
            covered.add((finding.technology, finding.criterion))

    required = [
        (technology, scenario["id"])
        for technology in ("TurboQuant", "CXL-based")
        for scenario in scenarios
    ]
    return [item for item in required if item not in covered]


def _merge_evidence_cards(
    initial_cards: list[EvidenceCard],
    retry_cards: list[EvidenceCard],
) -> list[EvidenceCard]:
    """재평가 결과를 중복 없이 합치고 EvidenceCard ID를 다시 매긴다."""
    merged: list[EvidenceCard] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for card in [*initial_cards, *retry_cards]:
        key = (
            card.get("technology"),
            card.get("claim"),
            card.get("source_url"),
        )
        if key in seen:
            continue
        seen.add(key)
        copied_card = dict(card)
        technology_slug = str(card.get("technology", "general")).lower().replace(
            "-", "_"
        )
        copied_card["evidence_id"] = (
            f"cloud-domain-{technology_slug}-{len(merged) + 1:03d}"
        )
        merged.append(copied_card)  # type: ignore[arg-type]
    return merged


def _format_missing_coverage(
    missing_coverage: list[tuple[str, str]],
) -> list[dict[str, str]]:
    """부족한 기술·시나리오를 payload에서 읽기 쉬운 형태로 바꾼다."""
    scenario_by_id = {scenario["id"]: scenario for scenario in CLOUD_SCENARIOS}
    return [
        {
            "technology": technology,
            "criterion": criterion,
            "scenario": scenario_by_id.get(criterion, {}).get("label", criterion),
        }
        for technology, criterion in missing_coverage
    ]


def _empty_result(
    *,
    status: Literal["needs_retry", "insufficient_evidence", "failed"],
    summary: str,
    queries: list[str],
    retry_queries: list[str],
    retry_count: int,
    limitations: list[str],
    errors: list[str],
) -> dict[str, Any]:
    """실패 경로에서도 완전한 AgentResult 형식을 반환한다."""
    return {
        "cloud_domain_result": {
            "agent_name": "cloud_domain_evaluation",
            "status": status,
            "summary": summary,
            "evidence_ids": [],
            "limitations": limitations,
            "errors": errors,
            "payload": {
                "evaluation_criteria": CLOUD_EVALUATION_CRITERIA,
                "scenarios": CLOUD_SCENARIOS,
                "search_queries": queries,
                "retry_queries": retry_queries,
                "retry_count": retry_count,
                "missing_scenarios": _format_missing_coverage(
                    [
                        (technology, scenario["id"])
                        for technology in ("TurboQuant", "CXL-based")
                        for scenario in CLOUD_SCENARIOS
                    ]
                ),
                "comparison": {},
                "finding_count": 0,
                "web_source_count": 0,
                "vector_source_count": 0,
            },
        },
        "evidence_cards": [],
    }


def _generate_cloud_scenarios_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """B 노드: 요청에 적용할 클라우드 사용 시나리오를 생성한다."""
    return {"scenarios": _build_cloud_scenarios(state["global_state"])}


def _generate_search_queries_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """C 노드: 시나리오를 묶은 최초 검색 질문을 생성한다."""
    queries = _build_queries(state["global_state"])
    return {
        "initial_queries": queries,
        "active_queries": queries,
    }


def _retrieve_sources_node(state: CloudDomainGraphState) -> dict[str, Any]:
    """D 노드: 같은 질문으로 Tavily와 논문 Vector DB를 모두 조회한다."""
    active_queries = state.get("active_queries", [])
    new_web_results, search_errors = _search_cloud_sources(active_queries)
    new_vector_chunks, vector_errors = _search_vector_sources(active_queries)

    return {
        "web_results": _merge_web_results(
            state.get("web_results", []),
            new_web_results,
        ),
        "vector_chunks": _merge_vector_chunks(
            state.get("vector_chunks", []),
            new_vector_chunks,
        ),
        "errors": [
            *state.get("errors", []),
            *search_errors,
            *vector_errors,
        ],
    }


def _extract_related_evidence_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """E 노드: 검색 결과에서 내용이 있는 기술·사례만 추려낸다."""
    web_results = [
        result
        for result in state.get("web_results", [])
        if str(result.get("url", "")).strip()
        and str(result.get("content", "")).strip()
    ]
    vector_chunks = [
        chunk
        for chunk in state.get("vector_chunks", [])
        if str(chunk.get("content", "")).strip()
    ]
    technical_cards = state.get("technical_cards", [])
    return {
        "web_results": web_results,
        "vector_chunks": vector_chunks,
        "related_source_count": (
            len(technical_cards) + len(web_results) + len(vector_chunks)
        ),
    }


def _evaluate_domain_fit_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """F 노드: 관련 근거를 이용해 시나리오별 도메인 적합성을 평가한다."""
    scenarios = state.get("scenarios", CLOUD_SCENARIOS)
    web_results = state.get("web_results", [])
    vector_chunks = state.get("vector_chunks", [])

    # Tavily와 Vector DB가 모두 비었으면 LLM이 기술 근거만으로 과도하게
    # 추론하지 않도록 평가를 만들지 않고 모든 시나리오를 부족 상태로 남긴다.
    if not web_results and not vector_chunks:
        missing_coverage = [
            (technology, scenario["id"])
            for technology in ("TurboQuant", "CXL-based")
            for scenario in scenarios
        ]
        return {
            "missing_coverage": missing_coverage,
            "summary": "클라우드 평가에 사용할 웹·Vector DB 근거가 없습니다.",
        }

    required_coverage = (
        state.get("missing_coverage") if state.get("retry_count", 0) > 0 else None
    )
    try:
        extraction = _extract_cloud_assessment(
            state["global_state"],
            state.get("technical_cards", []),
            web_results,
            vector_chunks,
            required_coverage=required_coverage,
        )
        new_cards, skipped_findings = _build_evidence_cards(
            extraction,
            state.get("technical_cards", []),
            web_results,
            vector_chunks,
        )
    except Exception as error:  # noqa: BLE001
        return {
            "fatal_error": str(error),
            "errors": [*state.get("errors", []), str(error)],
        }

    evidence_cards = _merge_evidence_cards(
        state.get("evidence_cards", []),
        new_cards,
    )
    findings = [*state.get("findings", []), *extraction.findings]
    comparison = dict(state.get("comparison", {}))
    comparison.update(extraction.comparison)
    limitations = [*state.get("limitations", []), *extraction.limitations]
    if skipped_findings:
        limitations.append(
            "일부 주장은 입력 근거와 출처 URL이 연결되지 않아 제외했습니다."
        )

    return {
        "evidence_cards": evidence_cards,
        "findings": findings,
        "comparison": comparison,
        "summary": extraction.summary,
        "limitations": list(dict.fromkeys(limitations)),
        "skipped_findings": [
            *state.get("skipped_findings", []),
            *skipped_findings,
        ],
        "fatal_error": "",
    }


def _check_scenario_coverage_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """G 노드: 7개 시나리오가 두 기술 모두에서 평가됐는지 확인한다."""
    if state.get("fatal_error"):
        return {}
    missing_coverage = _missing_scenario_coverage(
        state.get("evidence_cards", []),
        state.get("scenarios", CLOUD_SCENARIOS),
        state.get("findings", []),
    )
    return {"missing_coverage": missing_coverage}


def _route_after_scenario_check(state: CloudDomainGraphState) -> str:
    """G 조건 분기: 완료면 K, 부족하면 H, 치명적 오류면 J로 이동한다."""
    if state.get("fatal_error"):
        return "record_limitations"
    if state.get("missing_coverage"):
        return "prepare_retry"
    return "return_domain_result"


def _prepare_retry_node(state: CloudDomainGraphState) -> dict[str, Any]:
    """H 노드: 근거가 부족한 시나리오만 대상으로 재검색 질문을 만든다."""
    retry_count = state.get("retry_count", 0)
    used_query_count = len(state.get("initial_queries", [])) + len(
        state.get("retry_queries", [])
    )
    retry_possible = (
        retry_count < MAX_RETRIES and used_query_count < MAX_SEARCH_QUERIES
    )
    if not retry_possible:
        return {"retry_possible": False}

    retry_query = _build_retry_query(state.get("missing_coverage", []))
    return {
        "active_queries": [retry_query],
        "retry_queries": [*state.get("retry_queries", []), retry_query],
        "retry_count": retry_count + 1,
        "retry_possible": True,
    }


def _check_retry_available_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """I 노드: H에서 계산한 재시도 가능 여부를 명시적으로 보존한다."""
    return {"retry_possible": bool(state.get("retry_possible", False))}


def _route_after_retry_check(state: CloudDomainGraphState) -> str:
    """I 조건 분기: 가능하면 D로 돌아가고 불가능하면 J로 이동한다."""
    return "retrieve_sources" if state.get("retry_possible") else "record_limitations"


def _record_limitations_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """J 노드: 재검색 후에도 남은 시나리오와 실행 오류를 한계로 기록한다."""
    limitations = list(state.get("limitations", []))
    missing_coverage = state.get("missing_coverage", [])
    if missing_coverage:
        missing_text = ", ".join(
            f"{technology}:{criterion}"
            for technology, criterion in missing_coverage
        )
        limitations.append(
            "최대 1회 재검색 후에도 근거가 부족한 시나리오: " + missing_text
        )
    if state.get("fatal_error"):
        limitations.append("클라우드 도메인 구조화 평가에 실패했습니다.")
    if state.get("errors"):
        limitations.append("일부 Tavily 또는 Vector DB 조회가 실패했습니다.")
    return {"limitations": list(dict.fromkeys(limitations))}


def _return_domain_result_node(
    state: CloudDomainGraphState,
) -> dict[str, Any]:
    """K 노드: 지정된 AgentResult와 EvidenceCard 형식으로 결과를 반환한다."""
    evidence_cards = state.get("evidence_cards", [])
    missing_coverage = state.get("missing_coverage", [])
    fatal_error = state.get("fatal_error", "")

    if fatal_error:
        status = "failed"
    elif evidence_cards and not missing_coverage:
        status = "ok"
    else:
        status = "insufficient_evidence"

    result = {
        "agent_name": "cloud_domain_evaluation",
        "status": status,
        "summary": state.get(
            "summary",
            "클라우드 도메인 평가를 완료하지 못했습니다.",
        ),
        "evidence_ids": [card["evidence_id"] for card in evidence_cards],
        "limitations": state.get("limitations", []),
        "errors": state.get("errors", []),
        "payload": {
            "evaluation_criteria": CLOUD_EVALUATION_CRITERIA,
            "scenarios": state.get("scenarios", CLOUD_SCENARIOS),
            "search_queries": state.get("initial_queries", []),
            "retry_queries": state.get("retry_queries", []),
            "retry_count": state.get("retry_count", 0),
            "missing_scenarios": _format_missing_coverage(missing_coverage),
            "comparison": state.get("comparison", {}),
            "finding_count": len(evidence_cards),
            "web_source_count": len(state.get("web_results", [])),
            "vector_source_count": len(state.get("vector_chunks", [])),
            "related_source_count": state.get("related_source_count", 0),
            "skipped_findings": state.get("skipped_findings", []),
        },
    }
    return {"cloud_domain_result": result}


def build_cloud_domain_graph():
    """고정된 B~K 도식을 LangGraph 서브그래프로 구성한다."""
    graph = StateGraph(CloudDomainGraphState)

    # 도식의 사각형 단계를 각각 독립 노드로 등록한다.
    graph.add_node("generate_scenarios", _generate_cloud_scenarios_node)  # B
    graph.add_node("generate_search_queries", _generate_search_queries_node)  # C
    graph.add_node("retrieve_sources", _retrieve_sources_node)  # D
    graph.add_node("extract_related_evidence", _extract_related_evidence_node)  # E
    graph.add_node("evaluate_domain_fit", _evaluate_domain_fit_node)  # F
    graph.add_node("check_scenario_coverage", _check_scenario_coverage_node)  # G
    graph.add_node("prepare_retry", _prepare_retry_node)  # H
    graph.add_node("check_retry_available", _check_retry_available_node)  # I
    graph.add_node("record_limitations", _record_limitations_node)  # J
    graph.add_node("return_domain_result", _return_domain_result_node)  # K

    graph.add_edge(START, "generate_scenarios")
    graph.add_edge("generate_scenarios", "generate_search_queries")
    graph.add_edge("generate_search_queries", "retrieve_sources")
    graph.add_edge("retrieve_sources", "extract_related_evidence")
    graph.add_edge("extract_related_evidence", "evaluate_domain_fit")
    graph.add_edge("evaluate_domain_fit", "check_scenario_coverage")
    graph.add_conditional_edges(
        "check_scenario_coverage",
        _route_after_scenario_check,
        {
            "prepare_retry": "prepare_retry",
            "record_limitations": "record_limitations",
            "return_domain_result": "return_domain_result",
        },
    )
    graph.add_edge("prepare_retry", "check_retry_available")
    graph.add_conditional_edges(
        "check_retry_available",
        _route_after_retry_check,
        {
            "retrieve_sources": "retrieve_sources",
            "record_limitations": "record_limitations",
        },
    )
    graph.add_edge("record_limitations", "return_domain_result")
    graph.add_edge("return_domain_result", END)

    return graph.compile()


CLOUD_DOMAIN_GRAPH = build_cloud_domain_graph()


def cloud_domain_agent(state: GlobalState) -> dict[str, Any]:
    """A 요청을 받아 클라우드 LangGraph를 실행하고 K 결과만 GlobalState에 반환한다."""
    technical_cards = _technical_cards_from_state(state)
    technologies = {card.get("technology") for card in technical_cards}

    # 기술 조사 결과가 없으면 검색 이전의 필수 입력이 누락된 것이므로 공통 실패
    # 형식으로 반환한다. 정상 입력은 아래 LangGraph의 B 노드부터 실행된다.
    if not {"TurboQuant", "CXL-based"}.issubset(technologies):
        return _empty_result(
            status="insufficient_evidence",
            summary="두 기술의 기술 조사 근거가 모두 준비되지 않았습니다.",
            queries=_build_queries(state),
            retry_queries=[],
            retry_count=0,
            limitations=["TurboQuant와 CXL-based 기술 근거가 모두 필요합니다."],
            errors=[],
        )

    graph_result = CLOUD_DOMAIN_GRAPH.invoke(
        {
            "global_state": state,
            "technical_cards": technical_cards,
            "retry_queries": [],
            "retry_count": 0,
            "retry_possible": False,
            "web_results": [],
            "vector_chunks": [],
            "evidence_cards": [],
            "findings": [],
            "comparison": {},
            "missing_coverage": [],
            "limitations": [],
            "errors": [],
            "skipped_findings": [],
            "fatal_error": "",
        }
    )
    return {
        "cloud_domain_result": graph_result["cloud_domain_result"],
        "evidence_cards": graph_result.get("evidence_cards", []),
    }
