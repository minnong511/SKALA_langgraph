"""Shared research execution and provenance checks, adapted from legacy/research_base.

Agents receive tools through their context and return results; they never schedule
or call another agent. Source text is untrusted data, not executable instructions.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..schemas import (
    RESEARCH_AGENTS,
    AgentContext,
    AgentRequest,
    AgentResult,
    EvidenceCard,
    Finding,
    FollowUpRequest,
    SourceDocument,
)
from ..tools.retriever import split_text
from ..tools.web_search import canonical_url

PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"


def load_prompt(agent: str) -> str:
    return (PROMPT_DIR / f"{agent}.md").read_text(encoding="utf-8").strip()


def as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    return dict(value or {})


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=lambda x: as_dict(x), indent=2)


def normalized_text(value: str) -> str:
    """Ignore line-wrap and Unicode presentation differences, never paraphrase."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value)).strip()


def same_source_url(first: str, second: str) -> bool:
    try:
        return canonical_url(first) == canonical_url(second)
    except ValueError:
        return False


def source_material_count(sources) -> int:
    """Count documents separately from retrieved PDF page/chunk records."""
    identities = set()
    for source in sources:
        if source.file_path:
            identity = ("file", source.file_path)
        elif source.url:
            try:
                identity = ("url", canonical_url(source.url))
            except ValueError:
                identity = ("source", source.source_id)
        else:
            identity = ("source", source.source_id)
        identities.add(identity)
    return len(identities)


def quote_is_present(quote: str, content: str) -> bool:
    normalized_quote = normalized_text(quote)
    return bool(normalized_quote and normalized_quote in normalized_text(content))


def source_excerpts(source: SourceDocument) -> list[dict[str, str]]:
    """Offer immutable raw-text spans; the model selects rather than rewrites quotes."""
    return [
        {
            "excerpt_id": "X-"
            + hashlib.sha256(f"{source.source_id}:{start}:{end}:{text}".encode()).hexdigest()[:16],
            "text": text,
        }
        for start, end, text in split_text(source.content, 700, 100)
    ]


def research_sources(sources: dict[str, SourceDocument]) -> list[dict]:
    return [
        dict(source.model_dump(mode="json", exclude={"content"}), excerpts=source_excerpts(source))
        for source in sources.values()
    ]


def citation_diagnostics(output: ResearchOutput, sources: dict[str, SourceDocument]) -> list[dict]:
    records = []
    for card in output.evidence_cards:
        source = sources.get(card.source_id)
        quote = card.evidence_text
        own_match = bool(source and quote_is_present(quote, source.content))
        if source is None:
            category = "unknown_source"
        elif card.source_excerpt_id:
            category = (
                "excerpt_selected"
                if any(row["excerpt_id"] == card.source_excerpt_id for row in source_excerpts(source))
                else "unknown_excerpt"
            )
        elif own_match:
            category = "exact_quote"
        elif any(quote_is_present(quote, item.content) for item in sources.values()):
            category = "wrong_source"
        elif "..." in quote or "…" in quote:
            category = "noncontiguous_or_ellipsis"
        else:
            category = "unmatched_requires_review"
        records.append(
            {
                "evidence_id": card.evidence_id,
                "source_id": card.source_id,
                "source_excerpt_id": card.source_excerpt_id,
                "category": category,
                "proposed_quote": quote,
                "direct_quote_match": own_match,
            }
        )
    return records


class SearchQueryPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=6)
    rationale: str = ""


class ComparisonEntry(BaseModel):
    perspective: str
    turboquant: str
    itme: str
    conflict_or_condition: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: str = "unknown"


class SynthesisItem(BaseModel):
    perspective: str
    statement: str
    reason: str = ""
    condition: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: str = "unknown"


class SynthesisOutput(BaseModel):
    comparison_matrix: list[ComparisonEntry] = Field(default_factory=list)
    agreements: list[SynthesisItem] = Field(default_factory=list)
    conflicts: list[SynthesisItem] = Field(default_factory=list)
    conditional_findings: list[SynthesisItem] = Field(default_factory=list)
    evidence_limitations: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    neutral_conclusion: str
    conclusion_evidence_ids: list[str] = Field(default_factory=list)
    neutrality_check: str


class ReportOutput(BaseModel):
    markdown: str


class StructuredField(BaseModel):
    key: str
    value: str


class ResearchOutput(BaseModel):
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    structured_output: list[StructuredField] = Field(default_factory=list)


def result_for(
    request: AgentRequest, agent: str, *, status: str = "failed", summary: str, **values: Any
) -> AgentResult:
    return AgentResult(
        task_id=request.task_id,
        agent=agent,
        attempt=request.attempt,
        status=status,
        summary=summary,
        **values,
    )


def failure(request: AgentRequest, agent: str, message: str, **values: Any) -> AgentResult:
    return result_for(request, agent, summary=message, errors=[message], gaps=[message], **values)


def research_payload(request: AgentRequest, context: AgentContext) -> dict[str, Any]:
    technical = context.results.get("technical")
    return {
        "technologies": request.technologies,
        "domain": request.domain,
        "as_of_date": request.as_of_date.isoformat(),
        "questions": request.questions,
        "evaluation_criteria": request.context.get("evaluation_criteria", {}),
        "context": request.context,
        "technical_result": technical.model_dump(mode="json") if technical else None,
        "feedback": request.feedback,
    }


def collect_research(context: AgentContext) -> tuple[list[EvidenceCard], dict[str, SourceDocument]]:
    cards: list[EvidenceCard] = []
    sources: dict[str, SourceDocument] = {}
    for agent in RESEARCH_AGENTS:
        result = context.results.get(agent)
        if result:
            cards.extend(result.evidence_cards)
            for source in result.sources:
                if source.source_id not in sources:
                    sources[source.source_id] = source
    return cards, sources


def usable_evidence(
    context: AgentContext,
) -> tuple[list[EvidenceCard], dict[str, str], dict[str, SourceDocument]]:
    """Fail closed: an absent or conflicting verdict does not authorize a card."""
    cards, sources = collect_research(context)
    verification = context.results.get("verification")
    if verification is None or verification.status == "failed":
        return [], {}, sources
    verdicts: dict[str, str] = {}
    duplicate_verdicts: set[str] = set()
    for verdict in verification.verification:
        if verdict.evidence_id in verdicts:
            duplicate_verdicts.add(verdict.evidence_id)
        verdicts[verdict.evidence_id] = verdict.status
    counts: dict[str, int] = {}
    for card in cards:
        counts[card.evidence_id] = counts.get(card.evidence_id, 0) + 1
    allowed = [
        card
        for card in cards
        if counts[card.evidence_id] == 1
        and card.evidence_id not in duplicate_verdicts
        and verdicts.get(card.evidence_id) in {"verified", "uncertain"}
    ]
    return allowed, {card.evidence_id: verdicts[card.evidence_id] for card in allowed}, sources


def sanitize_research(
    output: ResearchOutput, sources: dict[str, SourceDocument], agent: str, reserved_ids: set[str]
) -> tuple[list[EvidenceCard], list[Finding], list[str]]:
    cards: list[EvidenceCard] = []
    gaps: list[str] = []
    seen: set[str] = set(reserved_ids)
    for original in output.evidence_cards:
        card = original.model_copy(deep=True)
        source = sources.get(card.source_id)
        excerpt_error = False
        if source is not None and card.source_excerpt_id:
            excerpt = next(
                (item for item in source_excerpts(source) if item["excerpt_id"] == card.source_excerpt_id),
                None,
            )
            if excerpt is None:
                excerpt_error = True
            else:
                card.evidence_text = excerpt["text"]
        reason = ""
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]*", card.evidence_id):
            reason = "인용에 사용할 수 없는 근거 ID"
        elif card.evidence_id in seen:
            reason = "중복된 근거 ID"
        elif source is None:
            reason = "실제 검색 결과에 없는 출처"
        elif excerpt_error:
            reason = "해당 출처에 없는 원문 발췌 ID"
        elif not quote_is_present(card.evidence_text, source.content):
            reason = "출처 원문에서 확인되지 않은 인용문"
        elif not card.claim.strip():
            reason = "빈 주장"
        elif card.source_url and not same_source_url(card.source_url, source.url):
            reason = "실제 출처와 다른 URL"
        elif card.source_file and card.source_file != source.file_path:
            reason = "실제 출처와 다른 파일 경로"
        if reason:
            gaps.append(f"{card.evidence_id}: {reason}")
            continue
        assert source is not None
        seen.add(card.evidence_id)
        card.source_agent = agent
        card.source_title, card.source_author = source.title, source.author
        card.source_url, card.source_file = source.url, source.file_path
        card.published_at, card.retrieved_at = source.published_at, source.accessed_at
        card.page_or_section = source.page_or_section
        for field in ("source_title", "source_author", "published_at", "retrieved_at", "page_or_section"):
            if not getattr(card, field):
                card.missing_metadata[field] = "제공된 원문 메타데이터에 없어 확인할 수 없음"
        if not (card.source_url or card.source_file):
            card.missing_metadata["source_location"] = "제공된 출처에 URL 또는 파일 위치가 없음"
        cards.append(card)
    ids = {card.evidence_id for card in cards}
    accepted_cards = cards
    while True:
        ids = {card.evidence_id for card in accepted_cards}
        rejected = {
            card.evidence_id
            for card in accepted_cards
            if card.evidence_id in card.supporting_evidence_ids
            or not set(card.supporting_evidence_ids) <= ids
        }
        if not rejected:
            break
        gaps.extend(
            f"{evidence_id}: 존재하지 않거나 자기 자신인 추론 연결 근거" for evidence_id in sorted(rejected)
        )
        accepted_cards = [card for card in accepted_cards if card.evidence_id not in rejected]
    ids = {card.evidence_id for card in accepted_cards}
    findings: list[Finding] = []
    for finding in output.findings:
        if not finding.evidence_ids or not set(finding.evidence_ids) <= ids:
            gaps.append(f"{finding.finding_id}: 실제 근거에 연결되지 않은 판단 제외")
        else:
            findings.append(finding)
    return accepted_cards, findings, gaps


def repair_source_links(
    output: ResearchOutput, sources: dict[str, SourceDocument]
) -> tuple[ResearchOutput, list]:
    """Repair only unambiguous exact excerpt links, never fuzzy-match claims or quotes."""
    fixed = output.model_copy(deep=True)
    catalog = {item["excerpt_id"]: source for source in sources.values() for item in source_excerpts(source)}
    repairs = []
    for card in fixed.evidence_cards:
        source = catalog.get(card.source_excerpt_id)
        if source is None:
            continue
        if (card.source_id, card.source_url, card.source_file) != (
            source.source_id,
            source.url,
            source.file_path,
        ):
            repairs.append(
                {
                    "evidence_id": card.evidence_id,
                    "old_source_id": card.source_id,
                    "source_id": source.source_id,
                    "method": "exact_excerpt_id",
                }
            )
        card.source_id, card.source_url, card.source_file = source.source_id, source.url, source.file_path
    return fixed, repairs


def run_research(
    request: AgentRequest,
    context: AgentContext,
    *,
    agent: str,
    default_queries: list[str],
    use_retriever: bool,
    use_web: bool,
) -> AgentResult:
    prompt = load_prompt(agent)
    payload = research_payload(request, context)
    tools = [
        tool
        for enabled, tool in ((use_retriever, context.retriever), (use_web, context.web_search))
        if enabled and tool is not None
    ]
    if not tools:
        return failure(request, agent, f"{agent}: 설정된 검색 도구가 없습니다.")
    sources: dict[str, SourceDocument] = {}
    read_pages: set[tuple] = set()
    citation_audit: list[dict] = []
    repair_used = False
    stop_reason = ""
    errors: list[str] = []
    history: list[dict[str, Any]] = []
    retry_feedback: list[str] = []
    final_output: ResearchOutput | None = None
    cards: list[EvidenceCard] = []
    findings: list[Finding] = []
    gaps: list[str] = []
    reserved_ids = {
        card.evidence_id
        for name, value in context.results.items()
        if name in RESEARCH_AGENTS and name != agent
        for card in value.evidence_cards
    }
    for search_round in range(request.limits.max_search_retries + 1):
        before_round = len(sources)
        if context.budget is not None and context.budget.research_exhausted:
            stop_reason = "후속 단계 호출 예산 보존"
            break
        context.emit(
            request,
            "research_search",
            "근거 조사",
            search_round=search_round + 1,
            technologies=request.technologies,
        )
        current_payload = {
            **payload,
            "missing_evidence_feedback": retry_feedback,
            "search_round": search_round + 1,
        }
        try:
            plan = context.ask(
                SearchQueryPlan, prompt + "\n분석 전 검증 가능한 검색 질문만 설계한다.", current_payload
            )
            queries = list(dict.fromkeys(query.strip() for query in plan.queries if query.strip()))
            if not queries:
                queries = default_queries
        except Exception as exc:
            queries = list(dict.fromkeys([*request.questions, *default_queries]))[:6]
            errors.append(f"검색 질문 생성 실패; 기본 질문 사용: {exc}")
        context.emit(
            request, "search_queries", "검색 질문 준비", queries=queries, search_round=search_round + 1
        )
        round_errors: list[str] = []
        for query in queries[: request.limits.research_query_limit]:
            for tool in tools:
                if len(sources) >= request.limits.research_source_limit or (
                    context.budget is not None and context.budget.research_exhausted
                ):
                    break
                try:
                    found = context.search(tool, query, request.limits.top_k)
                except Exception as exc:
                    context.emit(
                        request,
                        "search_error",
                        "검색 실패",
                        query=query,
                        reason=str(exc),
                        tool=type(tool).__name__,
                    )
                    round_errors.append(f"검색 실패 ({query}): {exc}")
                    continue
                before_count = len(sources)
                successful_reads = 0
                for document in found:
                    if (
                        successful_reads >= request.limits.research_read_limit
                        or len(sources) >= request.limits.research_source_limit
                    ):
                        break
                    if document.source_id in sources:
                        continue
                    page_key = (
                        document.file_path,
                        document.metadata.get("file_sha256"),
                        document.metadata.get("page"),
                    )
                    local_page = bool(all(page_key))
                    if local_page and page_key in read_pages:
                        continue
                    try:
                        document = context.read(document)
                        if not document.content.strip():
                            raise ValueError("원문 본문이 비어 있습니다.")
                        sources[document.source_id] = document
                        successful_reads += 1
                        if local_page:
                            read_pages.add(page_key)
                    except Exception as exc:
                        context.emit(
                            request,
                            "source_read_error",
                            "원문 확인 실패",
                            source_id=document.source_id,
                            reason=str(exc),
                        )
                        round_errors.append(f"원문 확인 실패 ({document.source_id}): {exc}")
                context.emit(
                    request,
                    "search_results",
                    "검색 결과 원문 확인 및 중복 제거",
                    query=query,
                    tool=type(tool).__name__,
                    returned=len(found),
                    added_chunks=len(sources) - before_count,
                    chunk_count=len(sources),
                    source_count=source_material_count(sources.values()),
                )
        errors.extend(round_errors)
        history.append(
            {
                "round": search_round + 1,
                "queries": queries,
                "source_count": source_material_count(sources.values()),
                "chunk_count": len(sources),
                "errors": round_errors,
            }
        )
        if search_round > 0 and len(sources) == before_round:
            stop_reason = "추가 검색에서 새 원문 없음, 확보한 결과와 한계 유지"
            break
        if not sources:
            retry_feedback = ["검색 가능한 원문을 확보하지 못했습니다.", *round_errors]
            continue
        analysis_payload = {
            **current_payload,
            "search_queries": queries,
            "sources": research_sources(sources),
            "evidence_id_prefix": f"{agent}-",
            "reserved_evidence_ids": sorted(reserved_ids),
        }
        try:
            final_output = context.ask(
                ResearchOutput,
                prompt + "\n이전 검증 피드백과 기술 조사 결과를 분석에 반영한다. "
                "각 근거 카드는 해당 source_id의 excerpts에서 source_excerpt_id를 반드시 선택한다. "
                "evidence_text는 선택한 발췌의 연속 원문만 복사한다. 번역, 요약, ... 결합은 금지한다. "
                "여러 발췌가 필요하면 근거 카드를 나누고 판단의 evidence_ids에 함께 연결한다. "
                "선택 발췌가 claim을 뒷받침하는지 확인한다.",
                analysis_payload,
            )
        except Exception as exc:
            errors.append(f"{agent} 분석 LLM 호출 실패: {exc}")
            break
        raw_diagnostics = citation_diagnostics(final_output, sources)
        final_output, repairs = repair_source_links(final_output, sources)
        cards, findings, rejected = sanitize_research(final_output, sources, agent, reserved_ids)
        citation_audit.append(
            {
                "round": search_round + 1,
                "cards": raw_diagnostics,
                "repairs": repairs,
                "rejected": rejected,
            }
        )
        if rejected and not repair_used and request.limits.max_search_retries > 0:
            repair_used = True
            try:
                repaired = context.ask(
                    ResearchOutput,
                    prompt + "\n새 검색 없이 기존 sources와 excerpts만 사용해 출처 연결을 수정한다. "
                    "source_excerpt_id를 정확히 선택하고, finding의 evidence_ids는 최종 카드 ID와 일치시킨다. "
                    "기존 유효 카드는 유지하고 수정 불가능한 항목은 제외한다.",
                    {
                        **analysis_payload,
                        "previous_output": final_output.model_dump(mode="json"),
                        "citation_errors": rejected,
                    },
                )
                repaired, repairs = repair_source_links(repaired, sources)
                new_cards, new_findings, new_rejected = sanitize_research(
                    repaired, sources, agent, reserved_ids
                )
                adopted = len(new_cards) >= len(cards) and len(new_findings) >= len(findings)
                citation_audit.append(
                    {
                        "round": search_round + 1,
                        "repair_only": True,
                        "adopted": adopted,
                        "cards": citation_diagnostics(repaired, sources),
                        "repairs": repairs,
                        "rejected": new_rejected,
                    }
                )
                if adopted:
                    final_output, cards, findings, rejected = repaired, new_cards, new_findings, new_rejected
            except Exception as exc:
                errors.append(f"기존 자료 인용 보정 실패: {type(exc).__name__}")
        gaps = list(dict.fromkeys([*final_output.missing_information, *rejected]))
        context.emit(
            request,
            "research_analysis",
            "근거 기반 분석 완료",
            evidence_count=len(cards),
            finding_count=len(findings),
            rejected_count=len(rejected),
            missing_information=final_output.missing_information,
        )
        if not cards:
            gaps.append("출처 본문과 연결된 근거 카드가 없습니다.")
        if not findings:
            gaps.append("검증 가능한 근거에 연결된 핵심 판단이 없습니다.")
        # Only missing evidence triggers another search; disclosed methodological
        # limitations remain in the result without forcing repetitive searches.
        # Citation formatting errors are not a reason to fetch more documents.
        retry_feedback = list(dict.fromkeys(final_output.missing_information))
        if cards and findings and search_round >= 1:
            stop_reason = "추가 조사 1회 완료, 남은 정보 부족은 보고서 한계로 유지"
            break
        if not retry_feedback:
            break
    if final_output is None:
        message = f"{agent}: 분석 가능한 원문 또는 분석 결과를 확보하지 못했습니다."
        return result_for(
            request,
            agent,
            summary=message,
            sources=list(sources.values()),
            gaps=[*retry_feedback, message],
            errors=errors or [message],
            data={"search_history": history, "citation_audit": citation_audit},
        )
    gaps = list(dict.fromkeys(gaps))
    if retry_feedback and len(history) >= request.limits.max_search_retries + 1:
        gaps.append("내부 검색 재시도 한도에 도달했습니다.")
    status = (
        "completed" if cards and findings and not gaps and not errors else ("partial" if cards else "failed")
    )
    follow_ups = (
        [FollowUpRequest(target_agent=agent, reason="근거 보완 필요", questions=retry_feedback)]
        if retry_feedback and not (cards and findings) and not stop_reason
        else []
    )
    return result_for(
        request,
        agent,
        status=status,
        summary=final_output.summary,
        findings=findings,
        evidence_cards=cards,
        sources=list(sources.values()),
        gaps=gaps,
        errors=errors,
        follow_up_requests=follow_ups,
        data={
            "search_history": history,
            "citation_audit": citation_audit,
            "search_stop_reason": stop_reason,
            "limitations": final_output.limitations,
            "structured_output": [field.model_dump() for field in final_output.structured_output],
        },
    )
