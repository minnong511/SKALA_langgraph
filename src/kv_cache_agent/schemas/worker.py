"""A worker receives a bounded assignment, not the entire mutable workflow."""

from typing import Literal

from pydantic import Field, model_validator

from kv_cache_agent.schemas.base import Contract
from kv_cache_agent.schemas.evidence import (
    ClaimAttribution,
    EvidenceCard,
    SourceRef,
    Technology,
)
from kv_cache_agent.schemas.report import ReportPlan, SectionDraft
from kv_cache_agent.schemas.research import ResearchTask, TaskResult


class WorkerInput(Contract):
    task: ResearchTask
    plan: ReportPlan
    source_refs: tuple[SourceRef, ...] = ()
    technical_results: tuple[TaskResult, ...] = ()
    existing_evidence: tuple[EvidenceCard, ...] = ()
    existing_drafts: tuple[SectionDraft, ...] = ()

    @model_validator(mode="after")
    def validate_assignment(self):
        self.task.validate_against(self.plan)
        if len(set(self.task.technologies)) != len(self.task.technologies):
            raise ValueError("Duplicate task technologies")
        if len(set(self.task.criteria)) != len(self.task.criteria):
            raise ValueError("Duplicate task criteria")
        specs = {s.section_id: s for s in self.plan.sections}
        for key in self.task.section_ids:
            spec = specs[key]
            if spec.perspective != self.task.agent:
                raise ValueError("Worker section has a different perspective")
            if spec.technologies and not set(spec.technologies).intersection(
                self.task.technologies
            ):
                raise ValueError("Task has no applicable section technology")
            if spec.criteria and not set(spec.criteria).intersection(
                self.task.criteria
            ):
                raise ValueError("Task has no applicable section criterion")
        for criterion in self.task.criteria:
            if not any(
                not specs[k].criteria or criterion in specs[k].criteria
                for k in self.task.section_ids
            ):
                raise ValueError("Task criterion is outside the requested sections")
        for technology in self.task.technologies:
            if not any(
                not specs[k].technologies or technology in specs[k].technologies
                for k in self.task.section_ids
            ):
                raise ValueError("Task technology is outside the requested sections")
        evidence_ids = [e.evidence_id for e in self.existing_evidence]
        if len(set(evidence_ids)) != len(evidence_ids):
            raise ValueError("Duplicate existing evidence")
        if set(evidence_ids) != set(self.task.existing_evidence_ids):
            raise ValueError("Existing evidence must match the assignment IDs")
        for result in self.technical_results:
            if result.agent != "technical" or result.plan_version != self.plan.version:
                raise ValueError("Cloud input must contain current technical results")
        if self.technical_results and self.task.agent != "cloud_domain":
            raise ValueError("Technical result dependencies belong to the cloud worker")
        for draft in self.existing_drafts:
            if (
                draft.section_id not in self.task.section_ids
                or draft.owner != self.task.agent
                or draft.plan_version != self.plan.version
            ):
                raise ValueError("Unowned or stale revision draft")
            if self.task.draft_versions.get(draft.section_id, 1) <= draft.version:
                raise ValueError("A revision must advance the draft version")
        return self

    def cells(self) -> tuple[tuple[str, Technology, str], ...]:
        specs = {s.section_id: s for s in self.plan.sections}
        return tuple(
            (key, technology, criterion)
            for key in self.task.section_ids
            for technology in self.task.technologies
            if not specs[key].technologies or technology in specs[key].technologies
            for criterion in self.task.criteria
            if not specs[key].criteria or criterion in specs[key].criteria
        )


class SourceQuote(Contract):
    source_id: str = Field(min_length=1)
    quote: str = ""
    passage_id: str | None = None

    @model_validator(mode="after")
    def original_reference_required(self):
        if not self.quote and not self.passage_id:
            raise ValueError(
                "A quotation requires an original passage ID or exact text"
            )
        return self


class WorkerFinding(Contract):
    section_id: str = Field(min_length=1)
    technology: Technology
    criterion: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: Literal["fact", "inference"]
    source_quotes: tuple[SourceQuote, ...] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    caveat: str
    attribution: ClaimAttribution | None


class WorkerGap(Contract):
    section_id: str
    technology: Technology
    criterion: str
    reason: str = Field(min_length=1)


class WorkerExtraction(Contract):
    findings: tuple[WorkerFinding, ...]
    gaps: tuple[WorkerGap, ...]
    limitations: tuple[str, ...]
