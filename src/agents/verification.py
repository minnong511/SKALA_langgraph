"""Reopen originals and combine deterministic provenance with semantic review."""

from __future__ import annotations

import re
from collections import Counter
from datetime import date
from email.utils import parsedate_to_datetime

from pydantic import BaseModel, Field

from ..schemas import (
    RESEARCH_AGENTS,
    AgentContext,
    AgentRequest,
    AgentResult,
    FollowUpRequest,
    VerificationVerdict,
)
from .base import (
    collect_research,
    failure,
    load_prompt,
    quote_is_present,
    result_for,
    same_source_url,
    source_material_count,
)


class VerificationOutput(BaseModel):
    summary: str
    verification: list[VerificationVerdict] = Field(default_factory=list)


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    cards, sources = collect_research(context)
    if not cards:
        return failure(request, "verification", "검증할 근거 카드가 없습니다.")
    counts = Counter(card.evidence_id for card in cards)
    prechecks: dict[str, list[str]] = {}
    hard_failures: set[str] = set()
    opened: dict[str, dict] = {}
    read_errors: dict[str, str] = {}
    context.emit(
        request,
        "verification_started",
        "근거 원문 재확인 시작",
        evidence_count=len(cards),
        source_count=source_material_count(sources.values()),
        chunk_count=len(sources),
    )
    for source_id, source in sources.items():
        if source_id not in {card.source_id for card in cards}:
            continue
        try:
            actual = context.read(source)
            if actual.source_id != source_id:
                raise ValueError("다시 연 원문의 출처 ID가 다릅니다.")
            if not actual.content.strip():
                raise ValueError("원문 본문이 비어 있습니다.")
            opened[source_id] = actual.model_dump(mode="json")
        except Exception as exc:
            read_errors[source_id] = f"원문 재확인 실패: {exc}"
    for card in cards:
        issues: list[str] = []
        actual = opened.get(card.source_id)
        if counts[card.evidence_id] > 1:
            issues.append("중복된 근거 ID")
            hard_failures.add(card.evidence_id)
        if actual is None:
            issues.append(read_errors.get(card.source_id, "출처 목록에 없는 source_id"))
            hard_failures.add(card.evidence_id)
        else:
            if not quote_is_present(card.evidence_text, actual["content"]):
                issues.append("근거 인용문이 실제 원문과 일치하지 않음")
                hard_failures.add(card.evidence_id)
            if card.source_url and not same_source_url(card.source_url, actual["url"]):
                issues.append("원문과 다른 source_url")
                hard_failures.add(card.evidence_id)
            if card.source_file and card.source_file != actual["file_path"]:
                issues.append("원문과 다른 source_file")
                hard_failures.add(card.evidence_id)
            if (
                card.page_or_section
                and card.page_or_section != actual["page_or_section"]
                and not quote_is_present(card.page_or_section, actual["content"])
            ):
                issues.append("원문에서 확인되지 않은 페이지 또는 섹션")
                hard_failures.add(card.evidence_id)
            if not (actual.get("url") or actual.get("file_path")):
                issues.append("다시 확인할 수 있는 출처 위치가 없음")
                hard_failures.add(card.evidence_id)
            if actual.get("published_at"):
                try:
                    try:
                        published_date = date.fromisoformat(actual["published_at"][:10])
                    except ValueError:
                        published_date = parsedate_to_datetime(actual["published_at"]).date()
                    if published_date > request.as_of_date:
                        issues.append("분석 기준일 이후에 발행된 출처")
                        hard_failures.add(card.evidence_id)
                except ValueError, TypeError, OverflowError:
                    issues.append("출처 발행일 형식을 확인할 수 없음")
        if not card.claim.strip():
            issues.append("빈 주장")
            hard_failures.add(card.evidence_id)
        if card.evidence_id in card.supporting_evidence_ids:
            issues.append("자기 자신을 추론 연결 근거로 사용함")
            hard_failures.add(card.evidence_id)
        if card.statement_type == "analysis_inference" and not card.supporting_evidence_ids:
            issues.append("분석 추론의 연결 근거 ID가 없음")
        if set(card.supporting_evidence_ids) - set(counts):
            issues.append("분석 추론이 존재하지 않는 근거 ID를 사용함")
            hard_failures.add(card.evidence_id)
        if (
            re.search(r"\d\s*(?:%|배|[xX]\b|bits?\b|ms\b|GB\b|tokens?\b)", card.claim)
            and not card.conditions.strip()
        ):
            issues.append("성능 수치의 baseline 및 실험 조건 누락")
        prechecks[card.evidence_id] = issues
    errors: list[str] = []
    try:
        output = context.ask(
            VerificationOutput,
            load_prompt("verification"),
            {
                "technologies": request.technologies,
                "domain": request.domain,
                "as_of_date": request.as_of_date.isoformat(),
                "questions": request.questions,
                "evaluation_criteria": request.context.get("evaluation_criteria", {}),
                "feedback": request.feedback,
                "evidence_cards": [card.model_dump(mode="json") for card in cards],
                "reopened_sources": list(opened.values()),
                "deterministic_prechecks": prechecks,
                "instruction": "모든 evidence_id에 정확히 하나의 verification 판정을 반환한다.",
            },
        )
    except Exception as exc:
        output = VerificationOutput(summary="근거 의미 검증 실패", verification=[])
        errors.append(f"근거 검증 LLM 호출 실패: {exc}")
    model_counts = Counter(item.evidence_id for item in output.verification)
    by_id = {item.evidence_id: item for item in output.verification}
    unknown_ids = set(by_id) - set(counts)
    if unknown_ids:
        errors.append("검증 모델이 입력에 없는 ID를 반환함: " + ", ".join(sorted(unknown_ids)))
    verdicts: list[VerificationVerdict] = []
    seen: set[str] = set()
    for card in cards:
        if card.evidence_id in seen:
            continue
        seen.add(card.evidence_id)
        candidate = by_id.get(card.evidence_id)
        reasons = list(prechecks[card.evidence_id])
        if card.evidence_id in hard_failures:
            status = "rejected"
        elif candidate is None or model_counts[card.evidence_id] != 1 or not candidate.reason.strip():
            status = "rejected"
            reasons.append("명시적이고 유일한 의미 검증 판정이 없어 사용 금지")
        elif reasons and candidate.status == "verified":
            status = "uncertain"
        else:
            status = candidate.status
        if candidate:
            reasons.append(candidate.reason)
        target = (
            card.source_agent
            if card.source_agent in RESEARCH_AGENTS
            else ("technical" if card.perspective in {"technical", "trl"} else card.perspective)
        )
        if target not in RESEARCH_AGENTS:
            target = "technical"
        verdicts.append(
            VerificationVerdict(
                evidence_id=card.evidence_id,
                status=status,
                reason="; ".join(dict.fromkeys(reasons)) or "검증 사유 누락",
                target_agent=target,
            )
        )
    # Iterate to a fixed point: rejecting an underlying card also rejects every
    # dependent inference, regardless of the input card order.
    by_verdict_id = {item.evidence_id: item for item in verdicts}
    for _ in range(len(verdicts)):
        changed = False
        for card in cards:
            if not card.supporting_evidence_ids:
                continue
            verdict = by_verdict_id[card.evidence_id]
            dependencies = [by_verdict_id.get(item) for item in card.supporting_evidence_ids]
            if (
                any(item is None or item.status == "rejected" for item in dependencies)
                and verdict.status != "rejected"
            ):
                verdict.status = "rejected"
                verdict.reason += "; 추론의 연결 근거가 제외됨"
                changed = True
            elif (
                any(item.status == "uncertain" for item in dependencies if item)
                and verdict.status == "verified"
            ):
                verdict.status = "uncertain"
                verdict.reason += "; 추론의 연결 근거가 불확실함"
                changed = True
        if not changed:
            break
    totals = Counter(item.status for item in verdicts)
    for index, verdict in enumerate(verdicts, start=1):
        context.emit(
            request,
            "verification_progress",
            f"근거 {index}/{len(verdicts)} 검증 완료",
            current=index,
            total=len(verdicts),
            evidence_id=verdict.evidence_id,
            verdict=verdict.status,
            reason=verdict.reason,
        )
    context.emit(
        request,
        "verification_completed",
        "근거 검증 결과 집계",
        total=len(verdicts),
        verified=totals["verified"],
        uncertain=totals["uncertain"],
        rejected=totals["rejected"],
    )
    follow_ups = [
        FollowUpRequest(
            target_agent=item.target_agent,
            reason=item.reason,
            questions=[item.reason],
            evidence_ids=[item.evidence_id],
        )
        for item in verdicts
        if item.status != "verified"
    ]
    passed = all(item.status == "verified" for item in verdicts) and not errors
    return result_for(
        request,
        "verification",
        status="failed" if errors and not output.verification else ("completed" if passed else "partial"),
        summary=output.summary,
        verification=verdicts,
        gaps=[f"{item.evidence_id}: {item.reason}" for item in verdicts if item.status != "verified"],
        follow_up_requests=follow_ups,
        errors=errors,
        sources=[sources[item] for item in opened],
        data={"passed": passed, "prechecks": prechecks, "reopened_source_ids": list(opened)},
    )
