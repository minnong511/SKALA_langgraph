"""Report-generation Agent for the final neutral evaluation report."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from state import AgentState

from .models import ReportOutput
from .utils import collect_evidence_cards, normalize_evidence_cards


SYSTEM_PROMPT = """
당신은 KV cache 다관점 평가 보고서 생성 에이전트다.
검증된 근거와 평가 종합 결과만 사용하고 새로운 사실이나 출처를 만들지 않는다.

필수 구조:
1. 맨 앞은 SUMMARY이며 핵심 결과를 반 페이지 이내로 요약한다.
2. 분석 배경과 문제 정의
3. 기술 선정과 선정 이유
4. 기술 개요와 공개 정보 기반 TRL
5. 시장·이해관계자·클라우드 도메인 평가
6. 관점 간 일치·상충과 시사점
7. 공개 정보, 실험 조건, 자료 부족 및 확증편향 방지 한계
8. 맨 마지막은 REFERENCE이며 실제 본문에서 사용한 자료만 적는다.

TurboQuant와 ITME의 단일 승자를 선언하지 않는다.
조건이 다른 논문의 성능 수치를 직접 비교하지 않는다.
불확실한 근거는 불확실하다고 표시하고 rejected 근거는 사용하지 않는다.
본문의 핵심 주장에는 대괄호 형태의 source_id를 연결한다.
""".strip()


PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        (
            "human",
            """
기술: {technologies}
도메인: {domain}
평가 기준: {evaluation_criteria}
기술 조사: {technical_result}
평가 종합: {synthesis_result}
근거 검증: {validation_result}
사용 가능한 근거: {evidence_cards}
출처 목록: {source_catalog}

이전 품질검토 수정 지시:
{revision_instructions}
""".strip(),
        ),
    ]
)


def build_report_agent_node(llm: Any) -> Callable[[AgentState], dict[str, Any]]:
    structured_llm = llm.with_structured_output(ReportOutput)

    def report_agent_node(state: AgentState) -> dict[str, Any]:
        validation = state.get("validation_result") or {}
        cards = normalize_evidence_cards(collect_evidence_cards(state))
        rejected_ids = set(validation.get("rejected_claim_ids", []))
        verified_ids = set(validation.get("verified_claim_ids", []))
        uncertain_ids = set(validation.get("uncertain_claim_ids", []))
        usable_ids = verified_ids | uncertain_ids

        usable_cards = [
            card
            for card in cards
            if card["claim_id"] not in rejected_ids
            and (not usable_ids or card["claim_id"] in usable_ids)
        ]
        revision_instructions = (state.get("report_quality") or {}).get(
            "revision_instructions", []
        )

        try:
            output = structured_llm.invoke(
                PROMPT.format_messages(
                    technologies=_json(state.get("technologies", {})),
                    domain=_json(state.get("domain", {})),
                    evaluation_criteria=_json(state.get("evaluation_criteria", {})),
                    technical_result=_json(state.get("technical_result", {})),
                    synthesis_result=_json(state.get("synthesis_result", {})),
                    validation_result=_json(validation),
                    evidence_cards=_json(usable_cards),
                    source_catalog=_json(state.get("source_catalog", [])),
                    revision_instructions=_json(revision_instructions),
                )
            )
            if isinstance(output, dict):
                output = ReportOutput.model_validate(output)
        except Exception as exc:
            message = f"보고서 생성 LLM 호출 실패: {exc}"
            return {
                "report_markdown": "",
                "reference_entries": [],
                "errors": [message],
            }

        report = output.markdown.strip()
        warnings = _structural_warnings(report)
        return {
            "report_markdown": report,
            "reference_entries": [
                reference.model_dump() for reference in output.reference_entries
            ],
            "warnings": warnings,
        }

    return report_agent_node


def _structural_warnings(report: str) -> list[str]:
    warnings: list[str] = []
    headings = [line.strip().lstrip("#").strip() for line in report.splitlines() if line.startswith("#")]
    if not headings or headings[0].upper() != "SUMMARY":
        warnings.append("보고서 첫 챕터가 SUMMARY가 아닙니다.")
    if not headings or headings[-1].upper() != "REFERENCE":
        warnings.append("보고서 마지막 챕터가 REFERENCE가 아닙니다.")
    return warnings


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
