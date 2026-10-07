import operator
from typing import Annotated, Literal, TypedDict

from kv_cache_agent.schemas.outputs import (
    AgentResult,
    EvidenceCard,
    ResearchPlan,
)


class ControlState(TypedDict, total=False):
    next_agent: str
    status: Literal[
        "planning",
        "researching",
        "verifying",
        "synthesizing",
        "writing",
        "completed",
        "failed",
    ]
    retry_count: dict[str, int]
    errors: list[str]


class GlobalState(TypedDict, total=False):
    user_query: str
    research_plan: ResearchPlan
    control: ControlState

    technical_result: AgentResult
    market_result: AgentResult
    stakeholder_result: AgentResult
    cloud_domain_result: AgentResult

    # 병렬로 실행되는 평가 에이전트가 이 필드에 근거 카드를 추가한다.
    evidence_cards: Annotated[list[EvidenceCard], operator.add]
    # 검증기가 확정한 카드만 별도 필드로 전달하여 종합 단계의 입력으로 사용한다.
    verified_evidence_cards: list[EvidenceCard]
    # 검증 완료와 부분 검증 카드를 합친 잠정 보고서 작성용 입력이다.
    usable_evidence_cards: list[EvidenceCard]

    verification_result: AgentResult
    synthesis_result: AgentResult
    final_report: str


# Foundation channels are additive until the five worker agents migrate in stages 6–10.
from kv_cache_agent.schemas.evidence import EvidenceCard as EvidenceRecord
from kv_cache_agent.schemas.evidence import VerificationDecision
from kv_cache_agent.schemas.report import ReportPlan, SectionDraft
from kv_cache_agent.schemas.research import CoverageItem, TaskResult, merge_versioned


class FoundationState(TypedDict, total=False):
    report_plan: ReportPlan
    task_results: Annotated[dict[str, TaskResult], merge_versioned]
    evidence_by_id: Annotated[dict[str, EvidenceRecord], merge_versioned]
    drafts_by_section: Annotated[dict[str, SectionDraft], merge_versioned]
    verification_decisions: dict[str, VerificationDecision]
    coverage: list[CoverageItem]
    active_task_ids: list[str]
    budget_usage: dict[str, int]
