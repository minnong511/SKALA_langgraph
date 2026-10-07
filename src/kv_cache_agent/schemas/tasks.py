"""Runtime planning contracts; perspectives do not determine task count."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Perspective = Literal[
    "technical_maturity",
    "market",
    "stakeholder",
    "domain_application",
    "risk",
    "additional",
]
REQUIRED_PERSPECTIVES = (
    "technical_maturity",
    "market",
    "stakeholder",
    "domain_application",
)
LEGACY_PERSPECTIVES = {
    "technical_maturity": "technical",
    "market": "market",
    "stakeholder": "stakeholder",
    "domain_application": "cloud_domain",
    "risk": "risk",
    "additional": "additional",
}


class SubTask(BaseModel):
    model_config = ConfigDict(extra="forbid")
    task_id: str = Field(min_length=1, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    perspective: Perspective
    technology: str | list[str]
    objective: str = Field(min_length=1, max_length=1500)
    query: str = Field(min_length=1, max_length=1000)
    search_queries: list[Annotated[str, Field(min_length=1, max_length=180)]] = Field(
        default_factory=list, max_length=3
    )
    preferred_domains: list[str] = Field(default_factory=list, max_length=6)
    expected_evidence: str = Field(default="", max_length=500)
    preferred_source: Literal["paper", "web", "hybrid"]
    priority: int = Field(ge=1, le=10)
    retry_count: int = Field(default=0, ge=0)

    @field_validator("technology")
    @classmethod
    def nonempty_technology(cls, value):
        items = [value] if isinstance(value, str) else value
        if not items or any(not item.strip() for item in items):
            raise ValueError("technology must contain nonempty names")
        return value

    @field_validator("objective", "query")
    @classmethod
    def nonempty_text(cls, value):
        if not value.strip():
            raise ValueError("task text must not be blank")
        return value.strip()

    @field_validator("search_queries")
    @classmethod
    def clean_search_queries(cls, value):
        if any(not q.strip() for q in value):
            raise ValueError("search queries must not be blank")
        return list(dict.fromkeys(q.strip() for q in value))

    @field_validator("preferred_domains")
    @classmethod
    def valid_domains(cls, value):
        import re

        if any(
            not re.fullmatch(r"(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}", domain)
            for domain in value
        ):
            raise ValueError("preferred domains must be domain names, not URLs")
        return list(dict.fromkeys(domain.lower() for domain in value))


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tasks: list[SubTask] = Field(min_length=1)
    planning_reason: str = Field(min_length=1, max_length=3000)

    @model_validator(mode="after")
    def unique_ids(self):
        ids = [task.task_id for task in self.tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task IDs must be unique within a plan")
        return self
