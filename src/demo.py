"""Deterministic graph fixtures: sample output is never real technology evidence."""

from datetime import datetime, timezone

from src.agents import supervisor
from src.schemas import (
    REPORT_HEADINGS,
    RESEARCH_AGENTS,
    AgentResult,
    EvidenceCard,
    Finding,
    SourceDocument,
    VerificationVerdict,
)

DEMO_NOTICE = "예시 데이터입니다. 실제 기술 조사 결과나 제출용 보고서가 아닙니다."


def research(request, context):
    name = context.agent
    cards, findings, sources = [], [], []
    for technology in request.technologies:
        index = len(cards) + 1
        source_id = f"demo-{name}-{index}"
        evidence_id = f"{name}-{request.attempt}-{index}"
        text = f"{technology}의 {name} 관점은 실제 원문과 적용 조건을 확인해야 한다. {DEMO_NOTICE}"
        sources.append(
            SourceDocument(
                source_id=source_id,
                title=f"합성 예시: {name} {technology}",
                content=text,
                source_type="demo",
                page_or_section="예시 1절",
                accessed_at=datetime.now(timezone.utc).isoformat(),
            )
        )
        cards.append(
            EvidenceCard(
                evidence_id=evidence_id,
                technology=technology,
                perspective=name,
                claim=text,
                source_id=source_id,
                evidence_text=text,
                source_title=sources[-1].title,
                source_agent=name,
                page_or_section="예시 1절",
                retrieved_at=sources[-1].accessed_at,
                missing_metadata={"published_at": "합성 예시이므로 발행일 없음"},
                conditions="시연 전용",
                limitations=DEMO_NOTICE,
            )
        )
        findings.append(
            Finding(
                finding_id=f"f-{evidence_id}",
                technology=technology,
                perspective=name,
                statement=text,
                evidence_ids=[evidence_id],
                conditions="시연 전용",
                limitations=DEMO_NOTICE,
            )
        )
    return AgentResult(
        task_id=request.task_id,
        agent=name,
        attempt=request.attempt,
        status="completed",
        summary=f"{name} 예시 결과 반환",
        findings=findings,
        evidence_cards=cards,
        sources=sources,
        data={"demo": True},
    )


def verification(request, context):
    cards = [card for name in RESEARCH_AGENTS for card in context.results[name].evidence_cards]
    verdicts = []
    for card in cards:
        verdicts.append(
            VerificationVerdict(
                evidence_id=card.evidence_id,
                status="verified",
                reason="합성 예시 내부의 인용 연결만 확인",
                target_agent=card.source_agent,
            )
        )
        context.emit(
            request,
            "verification_progress",
            f"근거 검증 {len(verdicts)}/{len(cards)}건",
            current=len(verdicts),
            total=len(cards),
        )
    return AgentResult(
        task_id=request.task_id,
        agent="verification",
        attempt=request.attempt,
        status="completed",
        summary="예시 근거 연결 확인",
        verification=verdicts,
        data={"demo": True},
    )


def synthesis(request, context):
    return AgentResult(
        task_id=request.task_id,
        agent="synthesis",
        attempt=request.attempt,
        status="completed",
        summary=DEMO_NOTICE,
        findings=[f for name in RESEARCH_AGENTS for f in context.results[name].findings],
        data={
            "demo": True,
            "neutral_conclusion": "기술의 승패를 정하지 않고 실제 자료 확보 후 조건별 평가가 필요합니다.",
        },
    )


def report(request, context):
    cards = [card for name in RESEARCH_AGENTS for card in context.results[name].evidence_cards]
    perspectives = "\n".join(f"- {card.claim} [{card.evidence_id}]" for card in cards)
    references = "\n".join(
        f"- [{card.evidence_id}] {card.source_title}. 합성 예시, 실제 출처 아님." for card in cards
    )
    sections = [
        DEMO_NOTICE,
        "클라우드 기반 LLM 서빙의 네 관점을 조사하는 흐름을 확인합니다.",
        ", ".join(request.technologies) + "를 비교 대상으로 입력했습니다.",
        "실제 원문에 근거한 기술 성숙도와 실험 조건의 확인이 필요합니다.",
        perspectives,
        "기술의 우열을 판정하지 않으며 적용 조건에 따라 검토합니다.",
        "모든 내용은 합성 예시입니다. 실제 성능, 시장 규모, 채택 사례, 발언은 검증하지 않았습니다.",
        references,
    ]
    markdown = (
        "\n\n".join(f"# {heading}\n\n{body}" for heading, body in zip(REPORT_HEADINGS, sections)) + "\n"
    )
    return AgentResult(
        task_id=request.task_id,
        agent="report",
        attempt=request.attempt,
        status="completed",
        summary="예시 보고서 작성",
        report_markdown=markdown,
        used_evidence_ids=[c.evidence_id for c in cards],
        data={"demo": True},
    )


def demo_agents():
    return {
        "supervisor": supervisor.run,
        **{name: research for name in RESEARCH_AGENTS},
        "verification": verification,
        "synthesis": synthesis,
        "report": report,
    }
