from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict as ValidatedTypedDict

from kv_cache_agent.schemas.tasks import Perspective


class EvidenceCard(ValidatedTypedDict, total=False):
    evidence_id: str
    technology: Literal["TurboQuant", "CXL-based", "both", "general"]
    perspective: Literal[
        "technical",
        "market",
        "stakeholder",
        "cloud_domain",
        "risk",
        "additional",
    ]
    task_id: str
    source_id: str
    metric: str | None
    value: str | None
    unit: str | None
    baseline: str | None
    conditions: dict[str, str | None]
    source_chunk_ids: list[str]
    claim: str
    evidence_text: str
    source_title: str
    source_url: str
    source_type: Literal["paper", "official", "news", "blog", "report"]
    source_locator: str
    retrieval_method: Literal["faiss", "tavily", "direct"]
    published_date: str
    claim_type: Literal["fact", "inference"]
    confidence: float
    caveat: str
    verification_status: Literal[
        "unverified",
        "verified",
        "partially_verified",
        "unsupported",
    ]


class ResearchPlan(TypedDict, total=False):
    technologies: list[str]
    perspectives: list[str]
    search_questions: dict[str, list[str]]


class AgentResult(TypedDict, total=False):
    agent_name: str
    status: Literal["ok", "needs_retry", "insufficient_evidence", "failed"]
    summary: str
    evidence_ids: list[str]
    limitations: list[str]
    errors: list[str]
    payload: dict[str, Any]


class WorkerResult(BaseModel):
    """Compact output without prompts, retrieved documents or raw responses."""

    model_config = ConfigDict(extra="forbid")
    task_id: str
    perspective: Perspective
    technology: str | list[str]
    status: Literal["success", "partial", "failed"]
    findings: list[str] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    retry_count: int = Field(default=0, ge=0)
    failure_kind: Literal[
        "transport", "insufficient_evidence", "extraction", "none"
    ] = "none"
    search_attempts: list[dict[str, Any]] = Field(default_factory=list)
