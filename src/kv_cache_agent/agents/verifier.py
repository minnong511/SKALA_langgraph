"""EvidenceCard를 원문과 대조하는 LangGraph 기반 근거 검증 에이전트."""

import re
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, TypedDict
from urllib.parse import urlparse

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pypdf import PdfReader

from kv_cache_agent.config import ROOT_DIR
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm
from kv_cache_agent.schemas.outputs import EvidenceCard
from kv_cache_agent.schemas.tool_outputs import FetchedSource
from kv_cache_agent.tools.source_fetcher import fetch_source

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "verifier.yaml"
MAX_RETRIES = 1
FRESHNESS_YEARS = 3
MAX_SOURCE_CHARS_PER_CARD = 8_000

SupportLevel = Literal["full", "partial", "none"]
ClaimTypeAssessment = Literal["correct", "should_be_inference", "unclear"]


class ClaimEvidenceDecision(BaseModel):
    """D 노드가 주장 한 건과 원문을 비교한 구조화 결과."""

    evidence_id: str
    support_level: SupportLevel
    matched_text: str = ""
    rationale: str
    claim_type_assessment: ClaimTypeAssessment


class ClaimComparisonBatch(BaseModel):
    """여러 EvidenceCard의 주장-원문 비교 결과."""

    decisions: list[ClaimEvidenceDecision] = Field(default_factory=list)


class VerificationGraphState(TypedDict, total=False):
    """검증 전용 LangGraph 노드 사이에서만 사용하는 LocalState."""

    global_state: GlobalState
    cards: list[EvidenceCard]
    active_evidence_ids: list[str]
    metadata_issues: dict[str, list[str]]
    source_documents: dict[str, FetchedSource]
    decisions: dict[str, dict[str, Any]]
    quality_checks: dict[str, dict[str, Any]]
    verified_cards: list[EvidenceCard]
    balance_result: dict[str, Any]
    verification_passed: bool
    retry_requests: list[dict[str, Any]]
    retry_count: int
    retry_possible: bool
    limitations: list[str]
    errors: list[str]
    tavily_fallback_ids: list[str]
    verification_result: dict[str, Any]


def _load_system_prompt() -> str:
    """검증 규칙을 YAML에서 읽어 Python 코드와 프롬프트를 분리한다."""
    with PROMPT_PATH.open(encoding="utf-8") as prompt_file:
        prompt_config = yaml.safe_load(prompt_file) or {}
    return str(prompt_config.get("system_prompt", ""))


def _card_id(card: EvidenceCard, index: int) -> str:
    """빈 ID도 검증 결과에서 추적할 수 있도록 임시 식별자를 만든다."""
    evidence_id = str(card.get("evidence_id", "")).strip()
    return evidence_id or f"missing-evidence-id-{index:03d}"


def _validate_card_metadata(card: EvidenceCard) -> list[str]:
    """B 단계에서 URL과 필수 메타데이터의 존재 및 기본 형식을 검사한다."""
    issues: list[str] = []
    required_text_fields = (
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
        "claim_type",
    )
    for field in required_text_fields:
        if not isinstance(card.get(field), str) or not str(card.get(field, "")).strip():
            issues.append(f"필수 메타데이터 누락: {field}")

    source_url = str(card.get("source_url", "")).strip()
    parsed = urlparse(source_url)
    is_web_url = parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    is_local_paper = card.get("source_type") == "paper" and bool(source_url)
    if source_url and not is_web_url and not is_local_paper:
        issues.append("지원하지 않는 출처 URL 형식")

    confidence = card.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        issues.append("confidence는 0 이상 1 이하의 숫자여야 함")

    if card.get("claim_type") not in {"fact", "inference"}:
        issues.append("claim_type은 fact 또는 inference여야 함")
    return issues


def _extract_pdf_pages(path: Path, source_locator: str) -> str:
    """논문 locator의 페이지를 우선 읽고 없으면 PDF 전체에서 일부를 읽는다."""
    reader = PdfReader(str(path))
    page_numbers = [int(number) for number in re.findall(r"\d+", source_locator)]
    if page_numbers:
        start = max(page_numbers[0] - 1, 0)
        end = min((page_numbers[-1] if len(page_numbers) > 1 else start + 1), len(reader.pages))
        selected_indexes = range(start, max(start + 1, end))
    else:
        selected_indexes = range(min(3, len(reader.pages)))

    page_texts = [
        (reader.pages[index].extract_text() or "").strip()
        for index in selected_indexes
        if 0 <= index < len(reader.pages)
    ]
    return "\n\n".join(text for text in page_texts if text)[:30_000]


def _fetch_original_source(card: EvidenceCard) -> FetchedSource:
    """C 단계에서 웹 URL 또는 로컬 논문 PDF의 원문을 다시 확인한다."""
    source_url = str(card.get("source_url", "")).strip()
    parsed = urlparse(source_url)
    if parsed.scheme in {"http", "https"}:
        return fetch_source(source_url)

    # 기술 조사 fixture의 논문 경로는 프로젝트 루트 기준 상대 경로다.
    source_path = Path(source_url)
    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path
    if not source_path.is_file() or source_path.suffix.lower() != ".pdf":
        return {
            "title": str(card.get("source_title", "")),
            "url": source_url,
            "content": "",
            "source_type": "pdf",
            "published_date": str(card.get("published_date", "")),
            "content_length": 0,
            "status_code": 0,
            "content_type": "application/pdf",
            "fetch_status": "error",
            "error": "로컬 논문 PDF를 찾을 수 없습니다.",
        }

    try:
        content = _extract_pdf_pages(
            source_path,
            str(card.get("source_locator", "")),
        )
        return {
            "title": str(card.get("source_title", "")),
            "url": source_url,
            "content": content,
            "source_type": "pdf",
            "published_date": str(card.get("published_date", "")),
            "content_length": len(content),
            "status_code": 200,
            "content_type": "application/pdf",
            "fetch_status": "ok" if content else "empty",
            "error": "",
        }
    except Exception as error:  # noqa: BLE001
        return {
            "title": str(card.get("source_title", "")),
            "url": source_url,
            "content": "",
            "source_type": "pdf",
            "published_date": str(card.get("published_date", "")),
            "content_length": 0,
            "status_code": 0,
            "content_type": "application/pdf",
            "fetch_status": "error",
            "error": f"논문 PDF 확인 실패: {error}",
        }


def _build_comparison_context(
    cards: list[EvidenceCard],
    sources: dict[str, FetchedSource],
) -> str:
    """D 노드의 LLM이 카드와 원문을 ID별로 정확히 비교할 문맥을 만든다."""
    context_parts: list[str] = []
    for index, card in enumerate(cards, start=1):
        evidence_id = _card_id(card, index)
        source = sources.get(evidence_id, {})
        content = str(source.get("content", ""))[:MAX_SOURCE_CHARS_PER_CARD]
        context_parts.append(
            "\n".join(
                [
                    f"[evidence_id={evidence_id}]",
                    f"technology={card.get('technology', '')}",
                    f"perspective={card.get('perspective', '')}",
                    f"claim_type={card.get('claim_type', '')}",
                    f"claim={card.get('claim', '')}",
                    f"submitted_evidence={card.get('evidence_text', '')}",
                    f"source_url={card.get('source_url', '')}",
                    f"source_locator={card.get('source_locator', '')}",
                    f"original_source={content}",
                ]
            )
        )
    return "\n\n---\n\n".join(context_parts)


def _compare_claims_with_sources(
    cards: list[EvidenceCard],
    sources: dict[str, FetchedSource],
) -> ClaimComparisonBatch:
    """D 단계에서 주장과 원문을 구조화 출력으로 비교한다."""
    llm = get_llm().with_structured_output(ClaimComparisonBatch)
    prompt = (
        "아래 EvidenceCard의 주장과 재확인한 원문을 비교하라.\n"
        "주장을 원문이 직접 뒷받침하면 full, 일부 조건이나 범위만 뒷받침하면 "
        "partial, 뒷받침하지 않으면 none으로 판정하라.\n"
        "원문이 직접 말하지 않은 해석을 fact로 표시했다면 "
        "claim_type_assessment를 should_be_inference로 판정하라.\n"
        "각 evidence_id를 빠짐없이 그대로 반환하라.\n\n"
        f"검증 입력:\n{_build_comparison_context(cards, sources)}"
    )
    response = llm.invoke(
        [
            SystemMessage(content=_load_system_prompt()),
            HumanMessage(content=prompt),
        ]
    )
    if isinstance(response, ClaimComparisonBatch):
        return response
    return ClaimComparisonBatch.model_validate(response)


def _parse_published_date(value: str) -> date | None:
    """YYYY-MM-DD 또는 YYYY-MM 형식의 발행일을 비교 가능한 날짜로 바꾼다."""
    clean_value = value.strip()
    for pattern in (r"^(\d{4})-(\d{2})-(\d{2})", r"^(\d{4})-(\d{2})"):
        match = re.match(pattern, clean_value)
        if not match:
            continue
        parts = [int(part) for part in match.groups()]
        if len(parts) == 2:
            parts.append(1)
        try:
            return date(parts[0], parts[1], parts[2])
        except ValueError:
            return None
    return None


def _source_quality(card: EvidenceCard) -> dict[str, Any]:
    """E 단계에서 출처 유형과 발행일을 기준으로 품질·최신성을 평가한다."""
    source_type = card.get("source_type")
    quality = {
        "paper": "high",
        "official": "high",
        "report": "medium",
        "news": "medium",
        "blog": "low",
    }.get(source_type, "low")

    published_date = _parse_published_date(str(card.get("published_date", "")))
    if published_date is None:
        freshness = "unknown"
    else:
        age_years = (datetime.now(tz=UTC).date() - published_date).days / 365.25
        freshness = "stale" if age_years > FRESHNESS_YEARS else "current"
    return {
        "quality": quality,
        "freshness": freshness,
        "published_date": str(card.get("published_date", "")),
    }


def _check_source_metadata_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """B 노드: 출처 URL과 필수 메타데이터를 확인한다."""
    issues: dict[str, list[str]] = {}
    seen_ids: set[str] = set()
    for index, card in enumerate(state.get("cards", []), start=1):
        evidence_id = _card_id(card, index)
        card_issues = _validate_card_metadata(card)
        if evidence_id in seen_ids:
            card_issues.append("중복 evidence_id")
        seen_ids.add(evidence_id)
        issues[evidence_id] = card_issues
    return {
        "metadata_issues": issues,
        "active_evidence_ids": list(issues),
    }


def _recheck_original_sources_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """C 노드: 활성 근거 카드의 웹 페이지 또는 논문 원문을 다시 가져온다."""
    cards = state.get("cards", [])
    active_ids = set(state.get("active_evidence_ids", []))
    sources = dict(state.get("source_documents", {}))
    limitations = list(state.get("limitations", []))
    errors = list(state.get("errors", []))
    tavily_fallback_ids = list(state.get("tavily_fallback_ids", []))
    source_cache: dict[str, FetchedSource] = {}

    for index, card in enumerate(cards, start=1):
        evidence_id = _card_id(card, index)
        if evidence_id not in active_ids:
            continue
        source_url = str(card.get("source_url", "")).strip()
        source = source_cache.get(source_url)
        if source is None:
            source = _fetch_original_source(card)
            source_cache[source_url] = source
        sources[evidence_id] = source
        if source.get("fetch_status") == "blocked":
            fallback_content = str(card.get("evidence_text", "")).strip()
            if card.get("retrieval_method") == "tavily" and fallback_content:
                # 원문 대신 Tavily 검색 요약만 사용하므로 부분 검증으로만 처리한다.
                sources[evidence_id] = {
                    **source,
                    "content": fallback_content,
                    "content_length": len(fallback_content),
                    "content_type": "text/plain; source=tavily",
                }
                if evidence_id not in tavily_fallback_ids:
                    tavily_fallback_ids.append(evidence_id)
                limitations.append(
                    f"{evidence_id}: 원문 접근 차단으로 Tavily 검색 요약만 사용했습니다."
                )
            else:
                errors.append(
                    f"{evidence_id} 원문 확인 실패: "
                    f"{source.get('error', source.get('fetch_status', 'error'))}"
                )
        elif source.get("fetch_status") != "ok":
            errors.append(
                f"{evidence_id} 원문 확인 실패: "
                f"{source.get('error', source.get('fetch_status', 'error'))}"
            )
    return {
        "source_documents": sources,
        "limitations": list(dict.fromkeys(limitations)),
        "errors": list(dict.fromkeys(errors)),
        "tavily_fallback_ids": tavily_fallback_ids,
    }


def _compare_claim_and_evidence_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """D 노드: 주장과 원문 근거 문장을 비교한다."""
    active_ids = set(state.get("active_evidence_ids", []))
    active_cards = [
        card
        for index, card in enumerate(state.get("cards", []), start=1)
        if _card_id(card, index) in active_ids
        and state.get("source_documents", {})
        .get(_card_id(card, index), {})
        .get("fetch_status")
        in {"ok", "blocked"}
        and bool(
            state.get("source_documents", {})
            .get(_card_id(card, index), {})
            .get("content")
        )
    ]
    decisions = dict(state.get("decisions", {}))
    errors = list(state.get("errors", []))

    if active_cards:
        try:
            batch = _compare_claims_with_sources(
                active_cards,
                state.get("source_documents", {}),
            )
            for decision in batch.decisions:
                if decision.evidence_id in active_ids:
                    decisions[decision.evidence_id] = decision.model_dump()
        except Exception as error:  # noqa: BLE001
            errors.append(f"주장-원문 비교 실패: {error}")

    # 원문을 못 가져왔거나 LLM이 누락한 카드는 지원되지 않는 근거로 남긴다.
    for evidence_id in active_ids:
        if evidence_id not in decisions:
            decisions[evidence_id] = {
                "evidence_id": evidence_id,
                "support_level": "none",
                "matched_text": "",
                "rationale": "원문 확인 또는 주장 비교 결과가 없습니다.",
                "claim_type_assessment": "unclear",
            }
    return {
        "decisions": decisions,
        "errors": list(dict.fromkeys(errors)),
    }


def _evaluate_source_quality_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """E 노드: 출처 품질과 발행일 최신성을 평가한다."""
    quality_checks: dict[str, dict[str, Any]] = {}
    for index, card in enumerate(state.get("cards", []), start=1):
        quality_checks[_card_id(card, index)] = _source_quality(card)
    return {"quality_checks": quality_checks}


def _classify_fact_and_inference_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """F 노드: 사실·추론 구분까지 반영해 카드별 최종 검증 상태를 정한다."""
    verified_cards: list[EvidenceCard] = []
    metadata_issues = state.get("metadata_issues", {})
    decisions = state.get("decisions", {})
    quality_checks = state.get("quality_checks", {})
    tavily_fallback_ids = set(state.get("tavily_fallback_ids", []))

    for index, original_card in enumerate(state.get("cards", []), start=1):
        evidence_id = _card_id(original_card, index)
        card = deepcopy(original_card)
        decision = decisions.get(evidence_id, {})
        quality = quality_checks.get(evidence_id, {})
        issues = metadata_issues.get(evidence_id, [])

        support_level = decision.get("support_level", "none")
        claim_type_assessment = decision.get("claim_type_assessment", "unclear")
        if issues or support_level == "none":
            verification_status = "unsupported"
        elif evidence_id in tavily_fallback_ids:
            # Tavily 검색 요약은 원문을 대체할 수 없으므로 검증 완료로 승격하지 않는다.
            verification_status = "partially_verified"
        elif support_level == "partial" or claim_type_assessment != "correct":
            verification_status = "partially_verified"
        else:
            verification_status = "verified"

        # 낮은 신뢰도의 블로그 또는 오래된 웹 자료는 직접 일치하더라도
        # 최신 클라우드 환경을 충분히 대표하지 못하므로 부분 검증으로 낮춘다.
        if verification_status == "verified" and quality.get("quality") == "low":
            verification_status = "partially_verified"
        if (
            verification_status == "verified"
            and quality.get("freshness") == "stale"
            and card.get("source_type") != "paper"
        ):
            verification_status = "partially_verified"

        card["verification_status"] = verification_status
        rationale = str(decision.get("rationale", "")).strip()
        caveats = [str(card.get("caveat", "")).strip(), rationale, *issues]
        if evidence_id in tavily_fallback_ids:
            caveats.append("원문 접근 차단으로 Tavily 검색 요약만 확인함")
        card["caveat"] = " | ".join(item for item in caveats if item)
        verified_cards.append(card)
    return {"verified_cards": verified_cards}


def _check_comparison_balance_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """G 노드: 두 기술이 같은 관점과 근거 수준으로 비교됐는지 확인한다."""
    usable_statuses = {"verified", "partially_verified"}
    technology_counts = {"TurboQuant": 0, "CXL-based": 0}
    perspective_counts = {
        "TurboQuant": set(),
        "CXL-based": set(),
    }

    for card in state.get("verified_cards", []):
        if card.get("verification_status") not in usable_statuses:
            continue
        technologies = (
            ("TurboQuant", "CXL-based")
            if card.get("technology") == "both"
            else (card.get("technology"),)
        )
        for technology in technologies:
            if technology not in technology_counts:
                continue
            technology_counts[technology] += 1
            perspective_counts[technology].add(str(card.get("perspective", "")))

    missing_technologies = [
        technology
        for technology, count in technology_counts.items()
        if count == 0
    ]
    shared_perspectives = perspective_counts["TurboQuant"] & perspective_counts[
        "CXL-based"
    ]
    balanced = not missing_technologies and bool(shared_perspectives)
    return {
        "balance_result": {
            "balanced": balanced,
            "technology_counts": technology_counts,
            "shared_perspectives": sorted(shared_perspectives),
            "missing_technologies": missing_technologies,
        }
    }


def _check_verification_pass_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """H 노드: 모든 카드와 기술 비교 균형이 검증을 통과했는지 확인한다."""
    cards = state.get("verified_cards", [])
    all_verified = bool(cards) and all(
        card.get("verification_status") == "verified" for card in cards
    )
    balanced = bool(state.get("balance_result", {}).get("balanced"))
    return {"verification_passed": all_verified and balanced}


def _route_after_verification_check(state: VerificationGraphState) -> str:
    """H 조건 분기: 통과하면 L, 실패하면 I로 이동한다."""
    return "return_verified_result" if state.get("verification_passed") else "request_revision"


def _request_revision_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """I 노드: 수정 또는 재조사가 필요한 카드와 사유를 정리한다."""
    retry_count = state.get("retry_count", 0)
    problematic_cards = [
        card
        for card in state.get("verified_cards", [])
        if card.get("verification_status") != "verified"
    ]
    active_ids = [str(card.get("evidence_id", "")) for card in problematic_cards]

    requests = list(state.get("retry_requests", []))
    for card in problematic_cards:
        requests.append(
            {
                "evidence_id": str(card.get("evidence_id", "")),
                "requested_action": "원문 재확인 또는 추가 근거 조사",
                "reason": str(card.get("caveat", "")),
            }
        )

    retry_possible = retry_count < MAX_RETRIES and bool(active_ids)
    return {
        "active_evidence_ids": active_ids,
        "retry_requests": requests,
        "retry_possible": retry_possible,
        "retry_count": retry_count + 1 if retry_possible else retry_count,
    }


def _check_reverification_available_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """J 노드: 재검증 가능 여부를 명시적으로 보존한다."""
    return {"retry_possible": bool(state.get("retry_possible", False))}


def _route_after_reverification_check(state: VerificationGraphState) -> str:
    """J 조건 분기: 가능하면 C로 돌아가고 불가능하면 K로 이동한다."""
    return "recheck_original_sources" if state.get("retry_possible") else "mark_uncertainty"


def _mark_uncertainty_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """K 노드: 재검증 후에도 남은 불확실성과 비교 불균형을 기록한다."""
    uncertain_ids = [
        str(card.get("evidence_id", ""))
        for card in state.get("verified_cards", [])
        if card.get("verification_status") != "verified"
    ]
    limitations = list(state.get("limitations", []))
    if uncertain_ids:
        limitations.append(
            "재검증 후에도 불확실한 근거: " + ", ".join(uncertain_ids)
        )
    if not state.get("balance_result", {}).get("balanced"):
        limitations.append("TurboQuant와 CXL-based 비교 근거의 균형이 부족합니다.")
    return {"limitations": list(dict.fromkeys(limitations))}


def _return_verified_result_node(
    state: VerificationGraphState,
) -> dict[str, Any]:
    """L 노드: 카드별 상태와 검증 세부 결과를 AgentResult로 반환한다."""
    cards = state.get("verified_cards", [])
    verified = [card for card in cards if card.get("verification_status") == "verified"]
    partial = [
        card
        for card in cards
        if card.get("verification_status") == "partially_verified"
    ]
    unsupported = [
        card for card in cards if card.get("verification_status") == "unsupported"
    ]
    balanced = bool(state.get("balance_result", {}).get("balanced"))
    status = "ok" if verified and balanced else "insufficient_evidence"

    result = {
        "agent_name": "evidence_verification",
        "status": status,
        "summary": (
            f"근거 {len(cards)}건 중 검증 {len(verified)}건, "
            f"부분 검증 {len(partial)}건, 미지원 {len(unsupported)}건입니다."
        ),
        "evidence_ids": [str(card.get("evidence_id", "")) for card in verified],
        "limitations": state.get("limitations", []),
        "errors": state.get("errors", []),
        "payload": {
            "verified_evidence_cards": verified,
            "partially_verified_cards": partial,
            "unsupported_cards": unsupported,
            "all_verified_cards": cards,
            "verification_details": state.get("decisions", {}),
            "source_quality": state.get("quality_checks", {}),
            "balance_result": state.get("balance_result", {}),
            "retry_requests": state.get("retry_requests", []),
            "retry_count": state.get("retry_count", 0),
            "tavily_fallback_ids": state.get("tavily_fallback_ids", []),
            "uncertain_evidence_ids": [
                str(card.get("evidence_id", ""))
                for card in [*partial, *unsupported]
            ],
        },
    }
    return {"verification_result": result}


def build_evidence_verification_graph():
    """고정된 B~L 검증 도식을 LangGraph 서브그래프로 구성한다."""
    graph = StateGraph(VerificationGraphState)

    graph.add_node("check_source_metadata", _check_source_metadata_node)  # B
    graph.add_node("recheck_original_sources", _recheck_original_sources_node)  # C
    graph.add_node("compare_claim_and_evidence", _compare_claim_and_evidence_node)  # D
    graph.add_node("evaluate_source_quality", _evaluate_source_quality_node)  # E
    graph.add_node("classify_fact_and_inference", _classify_fact_and_inference_node)  # F
    graph.add_node("check_comparison_balance", _check_comparison_balance_node)  # G
    graph.add_node("check_verification_pass", _check_verification_pass_node)  # H
    graph.add_node("request_revision", _request_revision_node)  # I
    graph.add_node("check_reverification_available", _check_reverification_available_node)  # J
    graph.add_node("mark_uncertainty", _mark_uncertainty_node)  # K
    graph.add_node("return_verified_result", _return_verified_result_node)  # L

    graph.add_edge(START, "check_source_metadata")
    graph.add_edge("check_source_metadata", "recheck_original_sources")
    graph.add_edge("recheck_original_sources", "compare_claim_and_evidence")
    graph.add_edge("compare_claim_and_evidence", "evaluate_source_quality")
    graph.add_edge("evaluate_source_quality", "classify_fact_and_inference")
    graph.add_edge("classify_fact_and_inference", "check_comparison_balance")
    graph.add_edge("check_comparison_balance", "check_verification_pass")
    graph.add_conditional_edges(
        "check_verification_pass",
        _route_after_verification_check,
        {
            "return_verified_result": "return_verified_result",
            "request_revision": "request_revision",
        },
    )
    graph.add_edge("request_revision", "check_reverification_available")
    graph.add_conditional_edges(
        "check_reverification_available",
        _route_after_reverification_check,
        {
            "recheck_original_sources": "recheck_original_sources",
            "mark_uncertainty": "mark_uncertainty",
        },
    )
    graph.add_edge("mark_uncertainty", "return_verified_result")
    graph.add_edge("return_verified_result", END)
    return graph.compile()


EVIDENCE_VERIFICATION_GRAPH = build_evidence_verification_graph()


def _empty_verification_result(summary: str) -> dict[str, Any]:
    """A 입력에 근거 카드가 없을 때도 완전한 AgentResult를 반환한다."""
    return {
        "verification_result": {
            "agent_name": "evidence_verification",
            "status": "insufficient_evidence",
            "summary": summary,
            "evidence_ids": [],
            "limitations": ["검증할 EvidenceCard가 필요합니다."],
            "errors": [],
            "payload": {
                "verified_evidence_cards": [],
                "partially_verified_cards": [],
                "unsupported_cards": [],
                "all_verified_cards": [],
                "verification_details": {},
                "source_quality": {},
                "balance_result": {},
                "retry_requests": [],
                "retry_count": 0,
                "tavily_fallback_ids": [],
                "uncertain_evidence_ids": [],
            },
        }
    }


def evidence_verification_agent(state: GlobalState) -> dict[str, Any]:
    """A 입력을 받아 검증 LangGraph를 실행하고 L 결과를 GlobalState에 반환한다."""
    cards = [deepcopy(card) for card in state.get("evidence_cards", [])]
    if not cards:
        return {
            **_empty_verification_result("검증할 근거 카드가 없습니다."),
            "verified_evidence_cards": [],
        }

    graph_result = EVIDENCE_VERIFICATION_GRAPH.invoke(
        {
            "global_state": state,
            "cards": cards,
            "active_evidence_ids": [],
            "metadata_issues": {},
            "source_documents": {},
            "decisions": {},
            "quality_checks": {},
            "verified_cards": [],
            "balance_result": {},
            "verification_passed": False,
            "retry_requests": [],
            "retry_count": 0,
            "retry_possible": False,
            "limitations": [],
            "errors": [],
            "tavily_fallback_ids": [],
        }
    )
    verification_result = graph_result["verification_result"]
    verified_cards = verification_result["payload"].get(
        "verified_evidence_cards", []
    )
    return {
        "verification_result": verification_result,
        "verified_evidence_cards": deepcopy(verified_cards),
    }
