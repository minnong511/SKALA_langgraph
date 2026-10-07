"""Source snapshots and decisions tied to the actual version of a claim."""

from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlparse

from pydantic import Field, model_validator

from kv_cache_agent.schemas.base import Contract, fingerprint

Technology = Literal["TurboQuant", "CXL-based", "both", "general"]
Perspective = Literal["technical", "market", "stakeholder", "cloud_domain"]
SourceType = Literal["paper", "official", "news", "report", "blog", "unknown"]
VerificationStatus = Literal[
    "unverified", "verified", "partially_verified", "unsupported"
]


class SourceRef(Contract):
    source_id: str = Field(min_length=1)
    location: str = Field(min_length=1)
    source_type: SourceType = "unknown"
    title: str = ""
    pages: tuple[int, ...] = ()
    chunk_ids: tuple[str, ...] = ()
    locator: str = ""
    character_range: tuple[int, int] | None = None
    version: str = ""
    published_date: str | None = None

    @model_validator(mode="after")
    def valid_location(self):
        parsed = urlparse(self.location)
        if parsed.scheme and parsed.scheme not in {"http", "https"}:
            raise ValueError("Source must be HTTP(S) or a local PDF")
        if parsed.scheme and not parsed.hostname:
            raise ValueError("Missing source hostname")
        if not parsed.scheme and not self.location.lower().endswith(".pdf"):
            raise ValueError("Local sources must be PDF files")
        if any(page < 1 for page in self.pages):
            raise ValueError("Pages are one-based")
        if len(set(self.pages)) != len(self.pages):
            raise ValueError("Duplicate source pages")
        if self.character_range is not None:
            start, end = self.character_range
            if start < 0 or end <= start or not self.version:
                raise ValueError(
                    "Character ranges require a valid span and source version"
                )
        return self


class SourceSnapshot(Contract):
    reference: SourceRef
    content: str = ""
    acquisition: Literal["tavily_extract", "local_pdf", "search_excerpt"]
    status: Literal["ok", "empty", "blocked", "error"]
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    content_hash: str = ""
    error: str = ""
    truncated: bool = False
    provider_request_id: str | None = None

    @model_validator(mode="after")
    def validate_snapshot(self):
        if self.retrieved_at.tzinfo is None:
            raise ValueError("Source timestamps must have a timezone")
        if self.status == "ok" and not self.content.strip():
            raise ValueError("Successful source snapshot requires content")
        actual = fingerprint(self.content)
        if self.content_hash and self.content_hash != actual:
            raise ValueError("Source content hash mismatch")
        object.__setattr__(self, "content_hash", actual)
        return self

    @property
    def verification_version(self) -> str:
        return fingerprint(
            {
                "reference": self.reference,
                "content_hash": self.content_hash,
                "status": self.status,
                "acquisition": self.acquisition,
                "truncated": self.truncated,
            }
        )


class ClaimAttribution(Contract):
    actor: str = Field(min_length=1)
    actor_group: str = Field(min_length=1)
    statement_kind: Literal["public_statement", "documented_fact", "analyst_inference"]
    position: Literal["positive", "negative", "mixed", "neutral", "uncertain"]


class EvidenceCard(Contract):
    evidence_id: str = Field(min_length=1)
    version: int = Field(default=1, ge=1)
    technology: Technology
    perspective: Perspective
    criterion: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    evidence_text: str = Field(min_length=1)
    source_refs: tuple[SourceRef, ...] = Field(min_length=1)
    claim_type: Literal["fact", "inference"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    caveat: str = ""
    attribution: ClaimAttribution | None = None


class VerificationDecision(Contract):
    claim_id: str
    claim_version: int = Field(ge=1)
    claim_hash: str
    source_versions: dict[str, str]
    policy_version: str
    status: VerificationStatus
    support_level: Literal["full", "partial", "none"]
    rationale: str
    matched_text: str = ""
    claim_type_assessment: Literal["correct", "should_be_inference", "unclear"]
    criterion_assessment: Literal["relevant", "irrelevant", "unclear"] = "unclear"
    issues: tuple[str, ...] = ()

    def applies_to(self, claim, snapshots: dict[str, SourceSnapshot]) -> bool:
        return (
            claim.claim_id == self.claim_id
            and claim.version == self.claim_version
            and fingerprint(claim) == self.claim_hash
            and all(
                key in snapshots and snapshots[key].verification_version == version
                for key, version in self.source_versions.items()
            )
        )
