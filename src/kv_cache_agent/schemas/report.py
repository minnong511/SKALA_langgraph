"""Stable section ownership and assertions present in report text."""

from typing import Literal

from pydantic import Field, model_validator

from kv_cache_agent.schemas.base import Contract
from kv_cache_agent.schemas.evidence import Perspective, Technology

Worker = Literal["technical", "market", "stakeholder", "cloud_domain"]


class SectionSpec(Contract):
    section_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    order: int = Field(ge=0)
    owner: Literal[
        "supervisor", "technical", "market", "stakeholder", "cloud_domain", "renderer"
    ]
    objective: str = Field(min_length=1)
    questions: tuple[str, ...] = ()
    technologies: tuple[Technology, ...] = ()
    perspective: Perspective | None = None
    criteria: tuple[str, ...] = ()
    direct_evidence_criteria: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    require_verified_facts: Literal[True] = True

    @model_validator(mode="after")
    def criterion_requirements(self):
        if not set(self.direct_evidence_criteria) <= set(self.criteria):
            raise ValueError("Direct evidence criterion is outside the section")
        if len(set(self.criteria)) != len(self.criteria) or any(
            not c.strip() for c in self.criteria
        ):
            raise ValueError("Invalid section criteria")
        return self


class ReportPlan(Contract):
    plan_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    user_query: str = Field(min_length=1)
    sections: tuple[SectionSpec, ...] = Field(min_length=1)
    technologies: tuple[Literal["TurboQuant", "CXL-based"], ...] = (
        "TurboQuant",
        "CXL-based",
    )
    required_perspectives: tuple[Perspective, ...] = (
        "technical",
        "market",
        "stakeholder",
        "cloud_domain",
    )

    @model_validator(mode="after")
    def validate_plan(self):
        ids = [s.section_id for s in self.sections]
        orders = [s.order for s in self.sections]
        if len(set(ids)) != len(ids) or len(set(orders)) != len(orders):
            raise ValueError("Section IDs and orders must be unique")
        if len(self.technologies) != 2 or set(self.technologies) != {
            "TurboQuant",
            "CXL-based",
        }:
            raise ValueError("Both project technologies are required")
        required = {"technical", "market", "stakeholder", "cloud_domain"}
        if (
            len(self.required_perspectives) != 4
            or set(self.required_perspectives) != required
        ):
            raise ValueError("Four evaluation perspectives are mandatory")
        if not required <= {s.perspective for s in self.sections}:
            raise ValueError("Missing evaluation sections")
        dependencies = {s.section_id: s.dependencies for s in self.sections}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(key):
            if key in visiting:
                raise ValueError("Cyclic section dependency")
            if key in visited:
                return
            visiting.add(key)
            for dependency in dependencies[key]:
                if dependency not in dependencies:
                    raise ValueError("Unknown section dependency")
                visit(dependency)
            visiting.remove(key)
            visited.add(key)

        for key in dependencies:
            visit(key)
        return self


class DraftClaim(Contract):
    claim_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    section_id: str = Field(min_length=1)
    technology: Technology
    perspective: Perspective
    criterion: str = Field(min_length=1)
    text: str = Field(min_length=1)
    claim_type: Literal["fact", "inference", "limitation"]
    evidence_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def evidence_for_fact(self):
        if self.claim_type == "fact" and not self.evidence_ids:
            raise ValueError("Facts require evidence IDs")
        if len(set(self.evidence_ids)) != len(self.evidence_ids):
            raise ValueError("Duplicate evidence IDs")
        return self


class SectionDraft(Contract):
    section_id: str
    plan_version: int = Field(ge=1)
    version: int = Field(default=1, ge=1)
    owner: Literal["supervisor", "technical", "market", "stakeholder", "cloud_domain"]
    claims: tuple[DraftClaim, ...] = ()
    limitations: tuple[str, ...] = ()
    status: Literal["draft", "needs_revision", "approved", "unavailable"] = "draft"

    @model_validator(mode="after")
    def section_claims(self):
        if any(c.section_id != self.section_id for c in self.claims):
            raise ValueError("Claim belongs to a different section")
        ids = [c.claim_id for c in self.claims]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate draft claims")
        return self


class ReportDocument(Contract):
    plan: ReportPlan
    sections: tuple[SectionDraft, ...]
    status: Literal["completed", "provisional", "failed"]
    termination_reason: str = ""

    @model_validator(mode="after")
    def document_sections(self):
        specs = {s.section_id: s for s in self.plan.sections}
        seen: set[str] = set()
        for draft in self.sections:
            if draft.section_id not in specs or draft.section_id in seen:
                raise ValueError("Unknown or duplicate report section")
            spec = specs[draft.section_id]
            if draft.owner != spec.owner or draft.plan_version != self.plan.version:
                raise ValueError("Wrong section owner or stale plan")
            seen.add(draft.section_id)
        required = {s.section_id for s in self.plan.sections if s.owner != "renderer"}
        if self.status == "completed" and (
            not required <= seen or any(s.status != "approved" for s in self.sections)
        ):
            raise ValueError("Completed report requires every section approved")
        return self
