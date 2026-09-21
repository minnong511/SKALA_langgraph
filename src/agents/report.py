"""Generate the specified report and rebuild references from actual citations."""

from __future__ import annotations

import re

from ..schemas import REPORT_HEADINGS, AgentContext, AgentRequest, AgentResult
from .base import ReportOutput, failure, load_prompt, result_for, usable_evidence

CITATION_PATTERN = re.compile(r"\[([^\]\n]+)\](?!\()")


def report_structure_errors(markdown: str) -> list[str]:
    headings = re.findall(r"^#{1,2}\s+(.+?)\s*#*\s*$", markdown, re.MULTILINE)
    if headings != list(REPORT_HEADINGS):
        return ["보고서 목차 또는 순서가 명세와 다릅니다: " + " → ".join(REPORT_HEADINGS)]
    positions = [
        re.search(r"^#{1,2}\s+" + re.escape(title) + r"\s*$", markdown, re.MULTILINE)
        for title in REPORT_HEADINGS
    ]
    errors: list[str] = []
    for current, following, title in zip(positions, positions[1:], REPORT_HEADINGS):
        if current and following and not markdown[current.end() : following.start()].strip():
            errors.append(f"내용이 없는 보고서 항목: {title}")
    return errors


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    synthesis = context.results.get("synthesis")
    if synthesis is None or synthesis.status == "failed":
        return failure(request, "report", "사용 가능한 평가 종합 결과가 없습니다.")
    cards, verdicts, sources = usable_evidence(context)
    if not cards:
        return failure(request, "report", "보고서에 사용할 검증 근거가 없습니다.")
    allowed = {card.evidence_id: card for card in cards}
    if not set(synthesis.used_evidence_ids) <= set(allowed):
        return failure(request, "report", "평가 종합 결과가 사용할 수 없는 근거를 인용합니다.")
    context.emit(
        request,
        "report_started",
        "보고서 생성 시작",
        headings=list(REPORT_HEADINGS),
        evidence_count=len(cards),
    )
    try:
        output = context.ask(
            ReportOutput,
            load_prompt("report"),
            {
                "technologies": request.technologies,
                "domain": request.domain,
                "as_of_date": request.as_of_date.isoformat(),
                "questions": request.questions,
                "evaluation_criteria": request.context.get("evaluation_criteria", {}),
                "synthesis_result": synthesis.model_dump(mode="json"),
                "evidence_cards": [
                    {**card.model_dump(mode="json"), "verification_status": verdicts[card.evidence_id]}
                    for card in cards
                ],
                "source_catalog": [
                    source.model_dump(mode="json")
                    for source in sources.values()
                    if source.source_id in {card.source_id for card in cards}
                ],
                "revision_instructions": request.feedback,
                "headings": list(REPORT_HEADINGS),
            },
        )
    except Exception as exc:
        return failure(request, "report", f"보고서 생성 LLM 호출 실패: {exc}")
    markdown = output.markdown.strip()
    issues = report_structure_errors(markdown)
    reference_heading = re.search(r"^#{1,2}\s+REFERENCE\s*$", markdown, re.MULTILINE)
    body = markdown[: reference_heading.start()].rstrip() if reference_heading else markdown
    cited = list(dict.fromkeys(CITATION_PATTERN.findall(body)))
    unknown = sorted(set(cited) - set(allowed))
    if unknown:
        issues.append("알 수 없거나 검증되지 않은 본문 인용: " + ", ".join(unknown))
    if not cited:
        issues.append("본문에 근거 ID 인용이 없습니다.")
    linked_urls = set(re.findall(r"\]\((https?://[^\s)]+)\)", body))
    known_urls = {card.source_url for card in cards if card.source_url}
    if linked_urls - known_urls:
        issues.append("본문에 실제 출처 목록에 없는 링크가 있습니다.")
    context.emit(
        request,
        "report_review",
        "보고서 목차 및 인용 검사",
        heading_count=len(re.findall(r"^#{1,2}\s+", markdown, re.MULTILINE)),
        citation_count=len(cited),
        unknown_citation_count=len(unknown),
        issues=issues,
    )
    if issues:
        return result_for(
            request,
            "report",
            status="failed",
            summary="보고서 검토 후 수정이 필요합니다.",
            gaps=issues,
            errors=issues,
            report_markdown=markdown,
            used_evidence_ids=[item for item in cited if item in allowed],
            data={"draft_markdown": markdown, "quality_issues": issues, "draft_requires_review": True},
        )
    # Ensure uncertainty is visible beside every citation, including the summary.
    for evidence_id in cited:
        if verdicts[evidence_id] == "uncertain":
            body = body.replace(f"[{evidence_id}]", f"(불확실한 근거) [{evidence_id}]")
    references: list[str] = []
    for evidence_id in cited:
        card = allowed[evidence_id]
        source = sources.get(card.source_id)
        if source is None:
            return failure(request, "report", f"인용 근거 {evidence_id}의 실제 출처가 없습니다.")
        title = source.title or source.source_id
        author = source.author or "저자 확인 불가"
        published = source.published_at or "발행일 확인 불가"
        location = source.url or source.file_path or "위치 확인 불가"
        locator = f" · {card.page_or_section}" if card.page_or_section else ""
        accessed = source.accessed_at or "조회일 확인 불가"
        references.append(
            f"- [{evidence_id}] {title} — {author} ({published}). {location}{locator}. 조회: {accessed}"
        )
    markdown = body + "\n\n## REFERENCE\n\n" + "\n".join(references) + "\n"
    gaps = list(synthesis.gaps)
    context.emit(
        request,
        "report_completed",
        "보고서 작성 완료",
        headings=list(REPORT_HEADINGS),
        citation_count=len(cited),
        reference_count=len(references),
    )
    return result_for(
        request,
        "report",
        status="partial" if gaps else "completed",
        summary="명세에 따른 보고서와 실제 본문 인용 출처를 생성했습니다.",
        report_markdown=markdown,
        used_evidence_ids=cited,
        gaps=list(dict.fromkeys(gaps)),
        data={
            "reference_entries": references,
            "quality_issues": [],
            "uncertain_evidence_ids": [item for item in cited if verdicts[item] == "uncertain"],
        },
    )
