"""LangGraph state: parallel workers exclusively update their own result fields."""

from typing import Any, TypedDict

from src.schemas import AgentRequest, AgentResult


class AgentState(TypedDict, total=False):
    request: AgentRequest
    config: dict[str, Any]
    plan: dict[str, Any]
    pending_tasks: list[str]
    technical_result: AgentResult
    market_result: AgentResult | None
    stakeholder_result: AgentResult | None
    domain_result: AgentResult | None
    verification_result: AgentResult | None
    synthesis_result: AgentResult | None
    report_result: AgentResult | None
    retry_counts: dict[str, int]
    attempts: dict[str, int]
    review_history: list[dict[str, Any]]
    status: str
    final_artifacts: dict[str, str]
    phase: str
    feedback: dict[str, list[str]]
    research_round: int
    budget_exhausted: bool
    review: dict[str, Any]
