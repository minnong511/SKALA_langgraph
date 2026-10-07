"""Checkpointable payload patches and single-writer control metadata."""

from typing import Annotated, Any, TypedDict

from kv_cache_agent.schemas.evaluation import QualityEvaluation
from kv_cache_agent.schemas.outputs import AgentResult, EvidenceCard, WorkerResult
from kv_cache_agent.schemas.tasks import ResearchPlan, SubTask


def _merge_by_id(left, right, key):
    # Associative upsert: replaying a completed task cannot append duplicates.
    items = {}
    for item in [*(left or []), *(right or [])]:
        data = item.model_dump() if hasattr(item, "model_dump") else item
        items[data[key]] = data
    return [items[item_id] for item_id in sorted(items)]


def merge_worker_results(left, right):
    return _merge_by_id(left, right, "task_id")


def merge_evidence_cards(left, right):
    return _merge_by_id(left, right, "evidence_id")


class PayloadState(TypedDict, total=False):
    user_query: str
    selected_technologies: list[str]
    target_domain: str
    research_plan: dict[str, Any] | ResearchPlan
    tasks: list[dict[str, Any] | SubTask]
    worker_results: Annotated[list[WorkerResult], merge_worker_results]
    evidence_cards: Annotated[list[EvidenceCard], merge_evidence_cards]
    verified_evidence_cards: list[EvidenceCard]
    usable_evidence_cards: list[EvidenceCard]
    verification: AgentResult
    synthesis: AgentResult
    report: str
    report_metrics: dict[str, int | None]
    evaluation: dict[str, Any] | QualityEvaluation | None
    limitations: list[str]


def merge_payload(left: PayloadState, right: PayloadState) -> PayloadState:
    """Explicitly reduce nested patches: LangGraph reduces the outer channel.

    Sequential nodes replace other fields; omitted fields remain intact.
    """
    merged = dict(left or {})
    for key, value in (right or {}).items():
        reducer = {
            "worker_results": merge_worker_results,
            "evidence_cards": merge_evidence_cards,
        }.get(key)
        merged[key] = reducer(merged.get(key, []), value) if reducer else value
    return merged


class ControlState(TypedDict, total=False):
    trace_id: str
    step_count: int
    revision_count: int
    planning_round: int
    task_status: dict[str, str]
    retry_count: dict[str, int]
    failed_tasks: list[str]
    last_error: str | None
    status: str
    decision: str
    decision_reason: str
    termination_reason: str | None
    limits: dict[str, int]


class GlobalState(TypedDict, total=False):
    payload: Annotated[PayloadState, merge_payload]
    control: ControlState
    # Input-only compatibility; initialize normalizes this into payload.
    user_query: str


class LegacyState(TypedDict, total=False):
    """Transient adapter input for preserved agents, never the parent State."""

    user_query: str
    research_plan: dict[str, Any]
    control: dict[str, Any]
    technical_result: AgentResult
    market_result: AgentResult
    stakeholder_result: AgentResult
    cloud_domain_result: AgentResult
    evidence_cards: list[EvidenceCard]
    verified_evidence_cards: list[EvidenceCard]
    usable_evidence_cards: list[EvidenceCard]
    verification_result: AgentResult
    synthesis_result: AgentResult
    final_report: str
    dynamic_worker_results: list[dict[str, Any]]
    quality_feedback: dict[str, Any]
