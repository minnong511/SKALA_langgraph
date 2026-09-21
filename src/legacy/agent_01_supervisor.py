"""Supervisor Agent: planning, retry coordination, and final quality review."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from state import AgentState

from .models import ReportQualityOutput, SupervisorPlanOutput


PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 KV cache 다관점 평가 워크플로의 Supervisor다.
TurboQuant와 CXL-Hybrid 메모리 기반 ITME를 클라우드 LLM 서빙 환경에서 평가한다.
기술 조사 이후 시장·이해관계자·도메인 평가를 병렬 수행하도록 계획한다.
각 에이전트의 책임이 중복되지 않게 하고, 근거 없는 우열 판정을 금지한다.
""".strip(),
        ),
        (
            "human",
            """
사용자 요청: {user_request}
기술: {technologies}
도메인: {domain}
평가 기준: {evaluation_criteria}
""".strip(),
        ),
    ]
)


QUALITY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
당신은 최종 보고서 품질 검토를 수행하는 Supervisor다.
보고서를 새로 작성하지 말고 문제와 수정 지시만 반환한다.

검사 기준:
- SUMMARY가 첫 챕터이고 반 페이지 이내의 핵심 결과 요약인지
- 기술 선정, 기술 개요, TRL, 시장, 이해관계자, 도메인, 시사점, 한계가 포함되는지
- REFERENCE가 마지막 챕터이고 실제 사용 자료만 포함하는지
- 주장과 인용이 연결되는지
- TurboQuant와 ITME의 승자를 단정하지 않는지
- 실험 조건이 다른 성능 수치를 직접 비교하지 않는지
- 공개 정보 기반 TRL 추정과 불확실성이 명시되는지
""".strip(),
        ),
        (
            "human",
            """
보고서:
{report}

근거 검증 결과:
{validation_result}

평가 종합 결과:
{synthesis_result}
""".strip(),
        ),
    ]
)


@dataclass
class SupervisorAgent:
    llm: Any

    def __post_init__(self) -> None:
        self._plan_model = self.llm.with_structured_output(SupervisorPlanOutput)
        self._quality_model = self.llm.with_structured_output(ReportQualityOutput)

    def plan_node(self, state: AgentState) -> dict[str, Any]:
        try:
            output = self._plan_model.invoke(
                PLAN_PROMPT.format_messages(
                    user_request=state.get("user_request", ""),
                    technologies=_json(state.get("technologies", {})),
                    domain=_json(state.get("domain", {})),
                    evaluation_criteria=_json(state.get("evaluation_criteria", {})),
                )
            )
            if isinstance(output, dict):
                output = SupervisorPlanOutput.model_validate(output)
            plan = output.model_dump()
            errors: list[str] = []
        except Exception as exc:
            plan = _fallback_plan()
            errors = [f"Supervisor 계획 LLM 호출 실패로 기본 계획 사용: {exc}"]

        return {
            "research_plan": plan,
            "agent_status": {
                "technical": "pending",
                "market": "pending",
                "stakeholder": "pending",
                "domain": "pending",
                "validation": "pending",
                "synthesis": "pending",
                "report": "pending",
            },
            "retry_counts": state.get(
                "retry_counts",
                {
                    "technical": 0,
                    "market": 0,
                    "stakeholder": 0,
                    "domain": 0,
                },
            ),
            "max_retries": state.get("max_retries", 2),
            "report_revision_count": state.get("report_revision_count", 0),
            "max_report_revisions": state.get("max_report_revisions", 2),
            "errors": errors,
        }

    def quality_node(self, state: AgentState) -> dict[str, Any]:
        report = state.get("report_markdown", "")
        if not report.strip():
            result = ReportQualityOutput(
                passed=False,
                missing_sections=["전체 보고서"],
                revision_instructions=["보고서 생성 Agent를 다시 실행하세요."],
            )
        else:
            try:
                result = self._quality_model.invoke(
                    QUALITY_PROMPT.format_messages(
                        report=report,
                        validation_result=_json(state.get("validation_result", {})),
                        synthesis_result=_json(state.get("synthesis_result", {})),
                    )
                )
                if isinstance(result, dict):
                    result = ReportQualityOutput.model_validate(result)
            except Exception as exc:
                message = f"보고서 품질 검토 LLM 호출 실패: {exc}"
                revision_count = state.get("report_revision_count", 0) + 1
                warnings = []
                if revision_count >= state.get("max_report_revisions", 2):
                    warnings.append("보고서 최대 수정 횟수에 도달했습니다.")
                return {
                    "report_quality": {
                        "passed": False,
                        "missing_sections": [],
                        "citation_errors": [],
                        "unsupported_claims": [],
                        "bias_flags": [],
                        "formatting_errors": [],
                        "revision_instructions": [message],
                    },
                    "report_quality_passed": False,
                    "report_revision_count": revision_count,
                    "errors": [message],
                    "warnings": warnings,
                }

        revision_count = state.get("report_revision_count", 0)
        warnings: list[str] = []
        if not result.passed:
            revision_count += 1
            if revision_count >= state.get("max_report_revisions", 2):
                warnings.append("보고서 최대 수정 횟수에 도달했습니다.")

        return {
            "report_quality": result.model_dump(),
            "report_quality_passed": result.passed,
            "report_revision_count": revision_count,
            "warnings": warnings,
        }


def build_supervisor_agent(llm: Any) -> SupervisorAgent:
    return SupervisorAgent(llm=llm)


def route_after_report_quality(state: AgentState) -> str:
    if state.get("report_quality_passed"):
        return "end"
    if state.get("report_revision_count", 0) < state.get("max_report_revisions", 2):
        return "revise"
    return "stop"


def _fallback_plan() -> dict[str, Any]:
    return {
        "objective": "TurboQuant와 ITME의 다관점 중립 평가",
        "tasks": [
            {
                "agent": "technical",
                "objective": "원문 기술·TRL 근거 추출",
                "required_outputs": ["technical_result", "technical_evidence_cards"],
                "tools": ["FAISS"],
            },
            {
                "agent": "market",
                "objective": "시장성·채택·생태계 조사",
                "required_outputs": ["market_result", "market_evidence_cards"],
                "tools": ["Tavily", "FAISS"],
            },
            {
                "agent": "stakeholder",
                "objective": "이해관계자별 반응과 장벽 조사",
                "required_outputs": [
                    "stakeholder_result",
                    "stakeholder_evidence_cards",
                ],
                "tools": ["Tavily"],
            },
            {
                "agent": "domain",
                "objective": "클라우드 LLM 서빙 적합성 평가",
                "required_outputs": ["domain_result", "domain_evidence_cards"],
                "tools": ["FAISS", "Tavily"],
            },
        ],
        "execution_order": [
            "technical",
            "market|stakeholder|domain",
            "validation",
            "synthesis",
            "report",
        ],
        "neutrality_rules": [
            "단일 승자를 정하지 않는다.",
            "실험 조건이 다른 수치를 직접 비교하지 않는다.",
            "사실, 저자 주장, 분석 추론을 구분한다.",
        ],
        "completion_criteria": [
            "4개 관점 평가 완료",
            "근거 검증 완료",
            "SUMMARY와 REFERENCE를 포함한 보고서 생성",
        ],
    }


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)
