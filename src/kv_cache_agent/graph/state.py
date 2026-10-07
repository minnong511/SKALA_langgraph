from typing import Literal, TypedDict

from kv_cache_agent.schemas.outputs import (
    AgentResult,
    EvidenceCard,
    ResearchPlan,
)
from kv_cache_agent.schemas.quality import ReportQualityResult


class ControlState(TypedDict, total=False):
    # Supervisor가 다음 호출 대상을 정한다. 하위 에이전트는 이 값을 정하지 않는다.
    next_agent: str
    status: Literal[
        "planning",
        "researching",
        "verifying",
        "synthesizing",
        "writing",
        "completed",
        "limited",
        "needs_review",
        "failed",
    ]
    step_count: int
    max_steps: int
    retry_count: dict[str, int]
    node_status: dict[str, str]
    gap_requests: dict[str, list[str]]
    routing_reason: str
    evidence_revision: int
    verified_revision: int
    evidence_ready: bool
    # 마지막 검수의 요약만 보존한다. 전체 결정 로그는 외부 JSONL/LangSmith에 둔다.
    evidence_review: dict[str, object]
    report_revision: int
    max_report_revisions: int
    errors: list[str]
    last_error: str | None


class GlobalState(TypedDict, total=False):
    user_query: str
    # 외부 로그와 LangSmith 실행을 이어 주는 상관 키.
    trace_id: str
    research_plan: ResearchPlan
    control: ControlState

    technical_result: AgentResult
    market_result: AgentResult
    stakeholder_result: AgentResult
    cloud_domain_result: AgentResult

    # Supervisor는 워커를 순차 호출한다. 재조사 시 같은 ID는 최신 카드로 교체한다.
    evidence_cards: list[EvidenceCard]
    # 검증기가 확정한 카드만 별도 필드로 전달하여 종합 단계의 입력으로 사용한다.
    verified_evidence_cards: list[EvidenceCard]
    # 검증 완료와 부분 검증 카드를 합친 잠정 보고서 작성용 입력이다.
    usable_evidence_cards: list[EvidenceCard]

    verification_result: AgentResult
    synthesis_result: AgentResult
    final_report: str
    quality_result: ReportQualityResult
    quality_feedback: list[str]
