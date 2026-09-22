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

    verification_result: AgentResult
    synthesis_result: AgentResult
    final_report: str

