"""기술 조사 결과를 GPT 구조화 출력으로 받기 위한 스키마."""

from typing import Literal

from pydantic import BaseModel, Field


class TechnicalFinding(BaseModel):
    """검색된 논문 chunk를 근거로 작성한 기술 주장."""

    technology: Literal["TurboQuant", "CXL-based", "both", "general"]
    claim: str
    evidence_text: str
    source_chunk_ids: list[str] = Field(min_length=1)
    claim_type: Literal["fact", "inference"]
    confidence: float = Field(ge=0.0, le=1.0)
    caveat: str = ""


class TechnicalExtraction(BaseModel):
    """기술 조사 에이전트의 구조화된 LLM 응답."""

    summary: str
    findings: list[TechnicalFinding]
    limitations: list[str]
