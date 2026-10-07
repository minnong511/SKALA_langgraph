"""Model proposals are distinct from authoritative, versioned dispatch commands."""

from typing import Literal

from pydantic import Field, model_validator

from kv_cache_agent.schemas.base import Contract
from kv_cache_agent.schemas.evidence import SourceRef
from kv_cache_agent.schemas.report import ReportPlan, SectionSpec, Worker
from kv_cache_agent.schemas.research import TaskResult


class ReportOutline(Contract):
    sections: tuple[SectionSpec, ...] = Field(min_length=1)


class Assignment(Contract):
    agent: Worker
    section_ids: tuple[str, ...] = Field(min_length=1)
    technologies: tuple[Literal["TurboQuant", "CXL-based"], ...] = Field(min_length=1)
    criteria: tuple[str, ...] = Field(min_length=1)
    objective: str = Field(min_length=1)
    questions: tuple[str, ...] = Field(min_length=1)
    feedback: tuple[str, ...] = ()
    action: Literal["research", "revise"] = "research"
    trigger: Literal[
        "missing", "verification_feedback", "reported_gap", "semantic_gap"
    ] = "missing"
    unanswered_questions: tuple[str, ...] = ()
    unanswered_question_ids: tuple[str, ...] = ()
    max_search_calls: int = Field(default=2, ge=0, le=3)


class RouteProposal(Contract):
    action: Literal["research", "revise", "finalize", "stop"]
    reason: str = Field(min_length=1)
    assignments: tuple[Assignment, ...] = ()

    @model_validator(mode="after")
    def action_payload(self):
        if (self.action in {"research", "revise"}) != bool(self.assignments):
            raise ValueError(
                "Dispatch proposal must have assignments; termination must not"
            )
        return self


class WorkflowInput(Contract):
    user_query: str = Field(min_length=1)
    report_plan: ReportPlan | None = None
    source_refs: tuple[SourceRef, ...] = ()


class WorkerDelivery(Contract):
    task_id: str
    plan_version: int
    round_id: int
    version: int = 1
    result: TaskResult
    issue_kind: Literal["none", "stale", "contract"] = "none"
    message: str = ""
