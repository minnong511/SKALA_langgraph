"""Neutral cross-perspective synthesis, using only explicitly reviewed evidence."""

from __future__ import annotations

import re

from ..schemas import RESEARCH_AGENTS, AgentContext, AgentRequest, AgentResult, FollowUpRequest
from .base import SynthesisOutput, failure, load_prompt, result_for, usable_evidence

SECTIONS = ("comparison_matrix", "agreements", "conflicts", "conditional_findings")


def run(request: AgentRequest, context: AgentContext) -> AgentResult:
    cards, verdicts, _ = usable_evidence(context)
    if not cards:
        return failure(
            request, "synthesis", "종합에 사용할 명시적으로 검증된 또는 불확실성 표시 근거가 없습니다."
        )
    context.emit(
        request,
        "synthesis_started",
        "검증 근거 평가 종합 시작",
        evidence_count=len(cards),
        perspectives=["trl", "market", "stakeholder", "domain"],
    )
    allowed = set(verdicts)
    safe_findings = {
        name: [
            finding.model_dump(mode="json")
            for finding in value.findings
            if finding.evidence_ids and set(finding.evidence_ids) <= allowed
        ]
        for name, value in context.results.items()
        if name in RESEARCH_AGENTS
    }
    prompt = load_prompt("synthesis") + "\n중립 결론의 근거는 conclusion_evidence_ids에도 반드시 기록한다."
    try:
        output = context.ask(
            SynthesisOutput,
            prompt,
            {
                "technologies": request.technologies,
                "domain": request.domain,
                "evaluation_criteria": request.context.get("evaluation_criteria", {}),
                "questions": request.questions,
                "feedback": request.feedback,
                "agent_findings": safe_findings,
                "usable_evidence": [
                    {**card.model_dump(mode="json"), "verification_status": verdicts[card.evidence_id]}
                    for card in cards
                ],
            },
        )
    except Exception as exc:
        return failure(request, "synthesis", f"평가 종합 LLM 호출 실패: {exc}")
    data = output.model_dump(mode="json")
    gaps = list(dict.fromkeys(output.coverage_gaps))
    used_ids: set[str] = set()
    for section in SECTIONS:
        safe_items: list[dict] = []
        for item in data[section]:
            ids = set(item["evidence_ids"])
            if not ids or not ids <= allowed:
                gaps.append(f"{section}: 미검증·제외·알 수 없는 근거를 인용한 판단 제외")
                continue
            inline_ids = set(
                re.findall(
                    r"\[([A-Za-z0-9_.:-]+)\]",
                    " ".join(str(v) for k, v in item.items() if k != "evidence_ids"),
                )
            )
            if not inline_ids <= ids:
                gaps.append(f"{section}: 선언된 근거와 본문 인용 불일치로 판단 제외")
                continue
            if any(verdicts[evidence_id] == "uncertain" for evidence_id in ids):
                item["confidence"] = "low"
                uncertainty = "불확실한 근거를 포함하므로 추가 확인이 필요함. "
                field = "conflict_or_condition" if section == "comparison_matrix" else "condition"
                item[field] = uncertainty + item.get(field, "")
            used_ids.update(ids)
            safe_items.append(item)
        data[section] = safe_items
    conclusion_ids = set(data["conclusion_evidence_ids"])
    inline_conclusion_ids = set(re.findall(r"\[([A-Za-z0-9_.:-]+)\]", data["neutral_conclusion"]))
    if not conclusion_ids or not conclusion_ids <= allowed or not inline_conclusion_ids <= conclusion_ids:
        data["neutral_conclusion"] = "근거와 연결되지 않은 종합 결론은 보류합니다."
        data["conclusion_evidence_ids"] = []
        gaps.append("종합 결론의 근거 연결 부족")
    else:
        used_ids.update(conclusion_ids)
        if any(verdicts[item] == "uncertain" for item in conclusion_ids):
            data["neutral_conclusion"] = "불확실한 근거를 포함한 조건부 판단: " + data["neutral_conclusion"]
    present = {item["perspective"] for item in data["comparison_matrix"]}
    missing_perspectives = {"trl", "market", "stakeholder", "domain"} - present
    gaps.extend(f"{perspective} 관점 비교 근거 부족" for perspective in sorted(missing_perspectives))
    data["coverage_gaps"] = list(dict.fromkeys(gaps))
    has_content = any(data[section] for section in SECTIONS)
    context.emit(
        request,
        "synthesis_completed",
        "관점별 평가 종합 완료",
        perspective_count=len(present),
        comparison_count=len(data["comparison_matrix"]),
        conflict_count=len(data["conflicts"]),
        agreement_count=len(data["agreements"]),
        evidence_count=len(used_ids),
        gap_count=len(gaps),
    )
    available_perspectives = {
        "trl" if card.perspective == "technical" else card.perspective for card in cards
    }
    missing_evidence = missing_perspectives - available_perspectives
    follow_ups = [
        FollowUpRequest(
            target_agent="technical" if perspective == "trl" else perspective,
            reason=f"{perspective} 관점의 검증된 근거가 없어 보완 필요",
            questions=[f"{perspective} 평가의 원문 근거와 적용 조건을 확보하세요."],
        )
        for perspective in sorted(missing_evidence)
    ]
    self_gaps = [
        gap
        for gap in data["coverage_gaps"]
        if gap not in {f"{perspective} 관점 비교 근거 부족" for perspective in missing_evidence}
    ]
    if self_gaps:
        follow_ups.append(
            FollowUpRequest(
                target_agent="synthesis", reason="근거 연결 또는 평가 구성 보완", questions=self_gaps
            )
        )
    return result_for(
        request,
        "synthesis",
        status=("partial" if gaps else "completed") if has_content else "failed",
        summary=data["neutral_conclusion"],
        data=data,
        gaps=data["coverage_gaps"],
        used_evidence_ids=sorted(used_ids),
        follow_up_requests=follow_ups,
    )
