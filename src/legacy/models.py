"""Structured LLM outputs used by Agent 5 and Agent 6."""

from typing import Literal

from pydantic import BaseModel, Field


AgentName = Literal["technical", "market", "stakeholder", "domain", "supervisor"]
Perspective = Literal[
    "technical",
    "trl",
    "market",
    "stakeholder",
    "domain",
    "cross_perspective",
]
Confidence = Literal["high", "medium", "low", "unknown"]


class ValidationIssue(BaseModel):
    claim_id: str = Field(description="문제가 발견된 주장 ID")
    issue_type: Literal[
        "missing_source",
        "broken_or_unverifiable_source",
        "claim_mismatch",
        "missing_experiment_condition",
        "low_quality_source",
        "outdated_source",
        "fact_inference_confusion",
        "unfair_comparison",
        "unsupported_claim",
    ]
    severity: Literal["warning", "error"]
    reason: str
    target_agent: AgentName
    required_action: str


class ValidationOutput(BaseModel):
    passed: bool = Field(
        description="치명적인 근거 오류 없이 종합 단계로 진행할 수 있는지 여부"
    )
    verified_claim_ids: list[str] = Field(default_factory=list)
    uncertain_claim_ids: list[str] = Field(default_factory=list)
    rejected_claim_ids: list[str] = Field(default_factory=list)
    issues: list[ValidationIssue] = Field(default_factory=list)
    summary: str


class ComparisonEntry(BaseModel):
    perspective: Perspective
    turboquant: str
    itme: str
    conflict_or_condition: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence


class SynthesisItem(BaseModel):
    perspective: Perspective
    statement: str
    reason: str
    condition: str
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: Confidence


class SynthesisOutput(BaseModel):
    comparison_matrix: list[ComparisonEntry] = Field(default_factory=list)
    agreements: list[SynthesisItem] = Field(default_factory=list)
    conflicts: list[SynthesisItem] = Field(default_factory=list)
    conditional_findings: list[SynthesisItem] = Field(default_factory=list)
    evidence_limitations: list[str] = Field(default_factory=list)
    coverage_gaps: list[str] = Field(default_factory=list)
    neutral_conclusion: str
    neutrality_check: str = Field(
        description="승자 판정이나 근거 없는 단정이 없음을 확인한 짧은 자체 점검"
    )


class SearchQueryPlan(BaseModel):
    queries: list[str] = Field(
        min_length=1,
        max_length=6,
        description="중복되지 않고 출처로 검증 가능한 검색 질문",
    )
    rationale: str


class FindingOutput(BaseModel):
    finding_id: str
    technology: Literal["TurboQuant", "ITME", "BOTH"]
    perspective: Perspective
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    conditions: str = ""
    limitation: str = ""
    confidence: Confidence = "unknown"


class EvidenceCardOutput(BaseModel):
    claim_id: str
    technology: Literal["TurboQuant", "ITME", "BOTH"]
    perspective: Perspective
    claim: str
    source_id: str
    source_title: str
    source_url: str = ""
    source_type: str
    published_at: str = ""
    page_or_section: str = ""
    evidence_text: str
    reported_value: str = ""
    baseline: str = ""
    experimental_context: str = ""
    limitation: str = ""
    statement_type: Literal[
        "unclassified",
        "verified_fact",
        "author_claim",
        "analysis_inference",
    ] = "unclassified"
    confidence: Confidence = "unknown"


class StructuredField(BaseModel):
    key: str
    value: str


class ResearchAgentOutput(BaseModel):
    summary: str
    findings: list[FindingOutput] = Field(default_factory=list)
    evidence_cards: list[EvidenceCardOutput] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    structured_output: list[StructuredField] = Field(default_factory=list)


class SupervisorTask(BaseModel):
    agent: Literal["technical", "market", "stakeholder", "domain"]
    objective: str
    required_outputs: list[str]
    tools: list[str]


class SupervisorPlanOutput(BaseModel):
    objective: str
    tasks: list[SupervisorTask]
    execution_order: list[str]
    neutrality_rules: list[str]
    completion_criteria: list[str]


class ReportQualityOutput(BaseModel):
    passed: bool
    missing_sections: list[str] = Field(default_factory=list)
    citation_errors: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    bias_flags: list[str] = Field(default_factory=list)
    formatting_errors: list[str] = Field(default_factory=list)
    revision_instructions: list[str] = Field(default_factory=list)


class ReferenceEntry(BaseModel):
    source_id: str
    reference_text: str


class ReportOutput(BaseModel):
    markdown: str = Field(
        description="SUMMARY로 시작하고 REFERENCE로 끝나는 전체 Markdown 보고서"
    )
    reference_entries: list[ReferenceEntry] = Field(default_factory=list)
