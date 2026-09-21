"""Shared requests, evidence, results, and injected runtime context."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from time import monotonic
from typing import Any, Literal, TypeVar
from uuid import uuid4

from pydantic import BaseModel, Field

from .common.source_cache import SourceCache

AgentName = Literal[
    "supervisor", "technical", "market", "stakeholder", "domain", "verification", "synthesis", "report"
]
Perspective = Literal["technical", "trl", "market", "stakeholder", "domain", "cross_perspective"]
RESEARCH_AGENTS = ("technical", "market", "stakeholder", "domain")
REPORT_HEADINGS = (
    "SUMMARY",
    "1. 분석 배경",
    "2. 기술 선정",
    "3. 기술 개요",
    "4. 관점별 평가",
    "5. 시사점",
    "6. 한계점",
    "REFERENCE",
)


class ExecutionLimits(BaseModel):
    max_search_retries: int = Field(default=2, ge=0, le=10)
    max_research_retries: int = Field(default=2, ge=0, le=10)
    max_report_revisions: int = Field(default=2, ge=0, le=10)
    max_synthesis_retries: int = Field(default=2, ge=0, le=10)
    max_total_calls: int = Field(default=400, ge=1)
    reserved_final_calls: int = Field(default=100, ge=0)
    research_query_limit: int = Field(default=3, ge=1, le=6)
    research_read_limit: int = Field(default=3, ge=1, le=20)
    research_source_limit: int = Field(default=12, ge=1)
    max_run_seconds: float = Field(default=1800, gt=0)
    top_k: int = Field(default=5, ge=1, le=20)


class AgentRequest(BaseModel):
    run_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    task_id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    technologies: list[str] = Field(default_factory=lambda: ["TurboQuant", "ITME"], min_length=1)
    domain: str = "클라우드 기반 LLM 서빙"
    as_of_date: date = Field(default_factory=date.today)
    questions: list[str] = Field(default_factory=list)
    context: dict[str, Any] = Field(default_factory=dict)
    feedback: list[str] = Field(default_factory=list)
    limits: ExecutionLimits = Field(default_factory=ExecutionLimits)
    attempt: int = Field(default=1, ge=1)


class SourceDocument(BaseModel):
    source_id: str
    title: str = ""
    content: str = ""
    url: str = ""
    file_path: str = ""
    author: str = ""
    source_type: str = "web"
    page_or_section: str = ""
    published_at: str = ""
    accessed_at: str = ""
    query: str = ""
    retrieval_method: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float | None = None


class EvidenceCard(BaseModel):
    evidence_id: str
    technology: str
    perspective: Perspective
    claim: str
    source_id: str
    evidence_text: str = Field(
        description="연속된 원문 그대로의 인용. 요약, 번역, 생략 기호로 문장 결합 금지"
    )
    source_excerpt_id: str = Field(default="", description="제공된 source의 excerpts 중 선택한 excerpt_id")
    source_title: str = ""
    source_author: str = ""
    source_url: str = ""
    source_file: str = ""
    page_or_section: str = ""
    published_at: str = ""
    retrieved_at: str = ""
    missing_metadata: dict[str, str] = Field(default_factory=dict)
    statement_type: Literal["fact", "author_claim", "analysis_inference"] = "author_claim"
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    conditions: str = ""
    limitations: str = ""
    source_agent: str = ""


class Finding(BaseModel):
    finding_id: str
    technology: str
    perspective: Perspective
    statement: str
    evidence_ids: list[str] = Field(default_factory=list)
    conditions: str = ""
    limitations: str = ""
    confidence: Literal["high", "medium", "low", "unknown"] = "unknown"


class FollowUpRequest(BaseModel):
    target_agent: AgentName
    reason: str
    questions: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class VerificationVerdict(BaseModel):
    evidence_id: str
    status: Literal["verified", "uncertain", "rejected"]
    reason: str
    target_agent: str = ""


class AgentResult(BaseModel):
    task_id: str
    agent: AgentName
    attempt: int = Field(default=1, ge=1)
    status: Literal["completed", "partial", "failed"]
    summary: str
    findings: list[Finding] = Field(default_factory=list)
    evidence_cards: list[EvidenceCard] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    follow_up_requests: list[FollowUpRequest] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    sources: list[SourceDocument] = Field(default_factory=list)
    verification: list[VerificationVerdict] = Field(default_factory=list)
    report_markdown: str = ""
    used_evidence_ids: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


T = TypeVar("T", bound=BaseModel)


@dataclass
class AgentContext:
    """Dependencies are injected; results are a snapshot of previously returned work."""

    llm: Any = None
    retriever: Any = None
    web_search: Any = None
    source_reader: Any = None
    events: Any = None
    budget: Any = None
    results: dict[str, AgentResult] = field(default_factory=dict)
    agent: str = ""
    demo: bool = False
    market_rag: bool = False
    stakeholder_rag: bool = False
    pdf_font_path: Path | None = None
    structured_output_method: str | None = None
    task_id: str = ""
    attempt: int = 1
    source_cache: Any = field(default_factory=SourceCache)

    @contextmanager
    def timed_call(self, kind: str, operation: str):
        if self.budget is not None:
            self.budget.consume(kind, research=self.agent in RESEARCH_AGENTS)
        call_id = uuid4().hex
        started = monotonic()
        status = "failed"
        details = dict(
            agent=self.agent,
            task_id=self.task_id,
            attempt=self.attempt,
            call_id=call_id,
            kind=kind,
            operation=operation,
        )
        if self.events is not None:
            self.events.emit("call_start", f"{kind} 호출 시작", **details)
        try:
            yield
            status = "completed"
        finally:
            if self.events is not None:
                self.events.emit(
                    "call_end",
                    f"{kind} 호출 종료",
                    **details,
                    status=status,
                    elapsed_seconds=round(monotonic() - started, 4),
                )

    def emit(self, request: AgentRequest, event: str, message: str, **details: Any) -> None:
        if self.events is not None:
            self.events.emit(
                event, message, task_id=request.task_id, agent=self.agent, attempt=request.attempt, **details
            )

    def ask(self, schema: type[T], system: str, payload: Any) -> T:
        if self.llm is None:
            raise RuntimeError("LLM이 설정되지 않았습니다.")
        messages = [("system", system), ("human", json.dumps(payload, ensure_ascii=False, default=str))]
        options = (
            {"method": self.structured_output_method, "strict": False}
            if self.structured_output_method
            else {}
        )
        with self.timed_call("llm", schema.__name__):
            result = self.llm.with_structured_output(schema, **options).invoke(messages)
            return result if isinstance(result, schema) else schema.model_validate(result)

    def search(self, tool: Any, query: str, limit: int = 5) -> list[SourceDocument]:
        if tool is None:
            raise RuntimeError("검색 도구가 설정되지 않았습니다.")
        with self.timed_call("search", type(tool).__name__):
            return [
                item if isinstance(item, SourceDocument) else SourceDocument.model_validate(item)
                for item in tool.search(query, limit=limit)
            ]

    def read(self, source: SourceDocument) -> SourceDocument:
        if self.source_reader is None:
            raise RuntimeError("원문 확인 도구가 설정되지 않았습니다.")

        def load():
            with self.timed_call("source_read", type(self.source_reader).__name__):
                value = self.source_reader.read(source)
                return value if isinstance(value, SourceDocument) else SourceDocument.model_validate(value)

        # Verification must independently reopen originals, including previously failed URLs.
        if self.agent not in RESEARCH_AGENTS:
            return load()
        from .tools.web_search import canonical_url

        key = (id(self.source_reader), source.source_id)
        local_path = source.metadata.get("local_path") or source.file_path
        if local_path:
            root = getattr(self.source_reader, "raw_dir", None)
            if root is not None:
                path = (Path(root) / local_path).resolve()
                stat = path.stat()
                key = (
                    id(self.source_reader),
                    str(path),
                    stat.st_mtime_ns,
                    stat.st_size,
                    source.metadata.get("file_sha256"),
                    source.metadata.get("page"),
                    source.page_or_section,
                )
        elif source.url:
            key = (
                id(self.source_reader),
                canonical_url(source.url),
                source.metadata.get("page"),
                source.page_or_section,
            )
        value, cached = self.source_cache.read(key, load)
        if cached and self.events is not None:
            self.events.emit(
                "source_cache_hit",
                "조사용 원문 재사용",
                agent=self.agent,
                task_id=self.task_id,
                attempt=self.attempt,
                source_id=source.source_id,
            )
        return value.model_copy(
            update={
                "source_id": source.source_id,
                "query": source.query,
                "metadata": {**source.metadata, **value.metadata},
            }
        )
