from typing import Any, Literal, TypedDict


class EvidenceCard(TypedDict, total=False):
    evidence_id: str
    technology: Literal["TurboQuant", "CXL-based", "both", "general"]
    perspective: Literal["technical", "market", "stakeholder", "cloud_domain"]
    claim: str
    evidence_text: str
    source_title: str
    source_url: str
    source_type: Literal["paper", "official", "news", "blog", "report"]
    source_locator: str
    retrieval_method: Literal["faiss", "tavily", "direct"]
    published_date: str
    # 서지 확장 필드는 선택 사항이다. 기존 공통 필드는 그대로 보존한다.
    authors: str
    publisher: str
    canonical_url: str
    retrieved_at: str
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
    report_outline: list[dict[str, str]]


class AgentResult(TypedDict, total=False):
    agent_name: str
    status: Literal["ok", "needs_retry", "insufficient_evidence", "failed"]
    summary: str
    evidence_ids: list[str]
    limitations: list[str]
    errors: list[str]
    payload: dict[str, Any]
