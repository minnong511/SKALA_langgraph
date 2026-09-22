"""TurboQuant와 CXL-based KV Cache의 클라우드 적용성을 평가하는 에이전트."""

from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
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


def cloud_domain_agent(state: GlobalState) -> dict[str, Any]:
    """두 KV Cache 기술을 클라우드 LLM 서빙 관점에서 평가한다."""
    # A → B: 요청을 받고 모든 요청에 적용할 주요 클라우드 시나리오를 생성한다.
    scenarios = _build_cloud_scenarios(state)

    # B → C: 시나리오를 묶어 최초 검색 질문을 2개까지 생성한다.
    initial_queries = _build_queries(state)
    retry_queries: list[str] = []
    retry_count = 0

    technical_cards = _technical_cards_from_state(state)
    technologies = {card.get("technology") for card in technical_cards}

    # 클라우드 비교의 공통 출발점인 두 기술의 논문 근거가 모두 있어야 한다.
    if not {"TurboQuant", "CXL-based"}.issubset(technologies):
        return _empty_result(
            status="insufficient_evidence",
            summary="두 기술의 기술 조사 근거가 모두 준비되지 않았습니다.",
            queries=initial_queries,
            retry_queries=retry_queries,
            retry_count=retry_count,
            limitations=["TurboQuant와 CXL-based 기술 근거가 모두 필요합니다."],
            errors=[],
        )

    # C → D: 같은 질문으로 Tavily와 논문 Vector DB를 모두 조회한다.
    web_results, search_errors = _search_cloud_sources(initial_queries)
    vector_chunks, vector_errors = _search_vector_sources(initial_queries)
    all_errors = [*search_errors, *vector_errors]

    # 검색 결과가 하나도 없으면 평가 전에 1회 재검색한다. 이 재검색까지 포함해
    # Tavily 질문 수는 최대 3개이므로 팀의 검색 제한도 지킨다.
    if not web_results and not vector_chunks and retry_count < MAX_RETRIES:
        all_coverage = [
            (technology, scenario["id"])
            for technology in ("TurboQuant", "CXL-based")
            for scenario in scenarios
        ]
        retry_query = _build_retry_query(all_coverage)
        retry_queries.append(retry_query)
        retry_count += 1
        retried_web, retry_search_errors = _search_cloud_sources([retry_query])
        retried_vectors, retry_vector_errors = _search_vector_sources([retry_query])
        web_results = _merge_web_results(web_results, retried_web)
        vector_chunks = _merge_vector_chunks(vector_chunks, retried_vectors)
        all_errors.extend(retry_search_errors)
        all_errors.extend(retry_vector_errors)

    if not web_results and not vector_chunks:
        return _empty_result(
            # 내부 재검색 1회를 이미 사용했으므로 더 이상 needs_retry로 돌리지 않고
            # 도식의 J 단계처럼 근거 부족과 평가 한계를 기록하여 반환한다.
            status="insufficient_evidence",
            summary="클라우드 평가에 사용할 웹·Vector DB 근거를 확보하지 못했습니다.",
            queries=initial_queries,
            retry_queries=retry_queries,
            retry_count=retry_count,
            limitations=[
                "최대 1회 재검색 후에도 클라우드 평가 근거를 확보하지 못했습니다."
            ],
            errors=all_errors,
        )

    # D → E → F: 검색 결과에서 기술·사례를 추출하고 시나리오별 적합성을 평가한다.
    try:
        extraction = _extract_cloud_assessment(
            state,
            technical_cards,
            web_results,
            vector_chunks,
        )
        evidence_cards, skipped_findings = _build_evidence_cards(
            extraction,
            technical_cards,
            web_results,
            vector_chunks,
        )
    except Exception as error:  # noqa: BLE001
        return _empty_result(
            status="failed",
            summary="클라우드 평가 결과를 생성하지 못했습니다.",
            queries=initial_queries,
            retry_queries=retry_queries,
            retry_count=retry_count,
            limitations=[],
            errors=[*all_errors, str(error)],
        )

    limitations = list(extraction.limitations)
    if skipped_findings:
        limitations.append("일부 주장은 입력 근거와 출처 URL이 연결되지 않아 제외했습니다.")

    # F → G: 7개 주요 시나리오가 두 기술 모두에 대해 평가됐는지 확인한다.
    all_findings = list(extraction.findings)
    missing_coverage = _missing_scenario_coverage(
        evidence_cards,
        scenarios,
        all_findings,
    )

    # G(아니오) → H → I(예) → D: 검색 예산이 남았을 때 부족한 시나리오만
    # 대상으로 한 질문을 만들어 Tavily와 Vector DB를 한 번 더 조회한다.
    if missing_coverage and retry_count < MAX_RETRIES:
        retry_query = _build_retry_query(missing_coverage)
        retry_queries.append(retry_query)
        retry_count += 1

        retried_web, retry_search_errors = _search_cloud_sources([retry_query])
        retried_vectors, retry_vector_errors = _search_vector_sources([retry_query])
        web_results = _merge_web_results(web_results, retried_web)
        vector_chunks = _merge_vector_chunks(vector_chunks, retried_vectors)
        all_errors.extend(retry_search_errors)
        all_errors.extend(retry_vector_errors)

        try:
            retry_extraction = _extract_cloud_assessment(
                state,
                technical_cards,
                web_results,
                vector_chunks,
                required_coverage=missing_coverage,
            )
            retry_cards, retry_skipped = _build_evidence_cards(
                retry_extraction,
                technical_cards,
                web_results,
                vector_chunks,
            )
            evidence_cards = _merge_evidence_cards(evidence_cards, retry_cards)
            all_findings.extend(retry_extraction.findings)
            skipped_findings.extend(retry_skipped)
            limitations.extend(retry_extraction.limitations)
            extraction.comparison.update(retry_extraction.comparison)
        except Exception as error:  # noqa: BLE001
            all_errors.append(f"부족한 시나리오 재평가 실패: {error}")

        missing_coverage = _missing_scenario_coverage(
            evidence_cards,
            scenarios,
            all_findings,
        )

    # I(아니오) → J: 1회 재검색 후에도 근거가 부족한 시나리오는 한계로 기록한다.
    if missing_coverage:
        missing_text = ", ".join(
            f"{technology}:{criterion}"
            for technology, criterion in missing_coverage
        )
        limitations.append(
            "최대 1회 재검색 후에도 근거가 부족한 시나리오: " + missing_text
        )

    if all_errors:
        limitations.append("일부 Tavily 또는 Vector DB 조회가 실패했습니다.")

    # G(예) 또는 J → K: 모든 경우 공통 AgentResult 형식으로 평가를 반환한다.
    status = "ok" if evidence_cards and not missing_coverage else "insufficient_evidence"

    return {
        "cloud_domain_result": {
            "agent_name": "cloud_domain_evaluation",
            "status": status,
            "summary": extraction.summary,
            "evidence_ids": [card["evidence_id"] for card in evidence_cards],
            "limitations": limitations,
            "errors": all_errors,
            "payload": {
                "evaluation_criteria": CLOUD_EVALUATION_CRITERIA,
                "scenarios": scenarios,
                "search_queries": initial_queries,
                "retry_queries": retry_queries,
                "retry_count": retry_count,
                "missing_scenarios": _format_missing_coverage(missing_coverage),
                "comparison": extraction.comparison,
                "finding_count": len(evidence_cards),
                "web_source_count": len(web_results),
                "vector_source_count": len(vector_chunks),
                "skipped_findings": skipped_findings,
            },
        },
        "evidence_cards": evidence_cards,
    }
