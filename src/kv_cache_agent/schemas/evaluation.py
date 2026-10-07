"""Hybrid evaluation output shared by the judge and conditional router."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class QualityEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    groundedness: bool
    neutrality: bool
    bias_control: bool
    perspective_coverage: bool
    groundedness_reason: str
    neutrality_reason: str
    bias_reason: str
    coverage_reason: str
    missing_perspectives: list[str] = Field(default_factory=list)
    missing_evidence_topics: list[str] = Field(default_factory=list)
    overall_pass: bool
    recommended_action: Literal[
        "pass",
        "additional_research",
        "resynthesis",
        "rewrite",
    ]
    rule_failures: list[str] = Field(default_factory=list)
