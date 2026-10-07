"""Three observed nodes shared by four workers; research gaps return to the caller."""

import json
from functools import lru_cache
from pathlib import Path
from typing import TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from kv_cache_agent.llm import get_llm
from kv_cache_agent.observability.logger import emit
from kv_cache_agent.observability.nodes import add_logged_node
from kv_cache_agent.schemas.base import stable_id
from kv_cache_agent.schemas.evidence import EvidenceCard
from kv_cache_agent.schemas.report import DraftClaim, SectionDraft, Worker
from kv_cache_agent.schemas.research import BudgetExceeded, BudgetLedger, TaskResult
from kv_cache_agent.schemas.worker import WorkerExtraction, WorkerInput
from kv_cache_agent.tools.research_sources import (
    SourceCollection,
    SourceCollector,
    TaskBudget,
    source_passages,
)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


class WorkerState(TypedDict, total=False):
    request: WorkerInput
    task: dict
    sources: SourceCollection
    extraction: WorkerExtraction
    errors: list[str]
    limitations: list[str]
    failed: bool
    status: str
    output: TaskResult


def _normalize(text):
    return " ".join(text.split()).casefold()


def _prompt(request, sources):
    sections = [
        section.model_dump(mode="json")
        for section in request.plan.sections
        if section.section_id in request.task.section_ids
    ]
    payload = {
        "user_query": request.plan.user_query,
        "assignment": request.task.model_dump(mode="json"),
        "sections": sections,
        "required_cells": request.cells(),
        "previous_drafts": [d.model_dump(mode="json") for d in request.existing_drafts],
        "technical_context_unverified": [
            {
                "task_id": result.task_id,
                "status": result.status,
                "claims": [c.text for d in result.drafts for c in d.claims],
            }
            for result in request.technical_results
        ],
        "original_sources": [
            {
                "source_id": s.reference.source_id,
                "reference": s.reference.model_dump(mode="json"),
                "passages": source_passages(s),
            }
            for s in sources.snapshots
        ],
    }
    return json.dumps(payload, ensure_ascii=False)


def _assemble(request, extraction, sources, errors, limitations, failed, usage):
    task = request.task
    cells = set(request.cells())
    available = {s.reference.source_id: s for s in sources.snapshots}
    passages = {
        source_id: {p["passage_id"]: p["text"] for p in source_passages(snapshot)}
        for source_id, snapshot in available.items()
    }
    cards = []
    claims = []
    missing = []
    seen = set()
    for finding in extraction.findings:
        cell = (finding.section_id, finding.technology, finding.criterion)
        problem = ""
        quotes = finding.source_quotes
        resolved_quotes = []
        for quote in quotes:
            text = quote.quote
            if quote.passage_id:
                text = passages.get(quote.source_id, {}).get(quote.passage_id, "")
            resolved_quotes.append((quote.source_id, text))
        if cell not in cells:
            problem = "Finding targets an unassigned section/technology/criterion"
        elif any(q.source_id not in available for q in quotes):
            problem = "Finding contains an unknown source reference"
        elif any(not text for _, text in resolved_quotes):
            problem = "Finding contains an unknown original passage reference"
        elif any(
            q.passage_id and q.quote and _normalize(q.quote) != _normalize(text)
            for q, (_, text) in zip(quotes, resolved_quotes, strict=True)
        ):
            problem = "Finding rewrote its selected original passage"
        elif any(
            _normalize(text) not in _normalize(available[source_id].content)
            for source_id, text in resolved_quotes
        ):
            problem = "Finding quotation does not occur in its original source"
        elif finding.claim_type == "inference" and not finding.caveat.strip():
            problem = "Inference requires its assumptions and limitations"
        elif task.agent == "stakeholder":
            attribution = finding.attribution
            if attribution is None:
                problem = (
                    "Stakeholder finding requires actor, statement kind and position"
                )
            elif (finding.claim_type == "inference") != (
                attribution.statement_kind == "analyst_inference"
            ):
                problem = "A predicted stakeholder reaction must be analyst inference"
            elif attribution.statement_kind == "public_statement" and not any(
                _normalize(attribution.actor)
                in _normalize(available[q.source_id].content)
                for q in quotes
            ):
                problem = "Public statement actor does not occur in the source body"
        if problem:
            missing.append(
                f"{finding.section_id}/{finding.technology}/{finding.criterion}: {problem}"
            )
            continue
        refs = tuple(dict.fromkeys(available[q.source_id].reference for q in quotes))
        evidence_id = stable_id(
            "evidence",
            {
                "plan_id": task.plan_id,
                "owner": task.agent,
                "cell": cell,
                "finding": finding.model_dump(mode="json"),
                "sources": [ref.model_dump(mode="json") for ref in refs],
            },
        )
        if evidence_id in seen:
            continue
        seen.add(evidence_id)
        version = task.draft_versions.get(finding.section_id, 1)
        card = EvidenceCard(
            evidence_id=evidence_id,
            technology=finding.technology,
            perspective=task.agent,
            criterion=finding.criterion,
            claim=finding.text,
            evidence_text="\n\n".join(text for _, text in resolved_quotes),
            source_refs=refs,
            claim_type=finding.claim_type,
            confidence=finding.confidence,
            caveat=finding.caveat,
            attribution=finding.attribution,
        )
        claim = DraftClaim(
            claim_id=stable_id(
                "claim", (task.plan_id, cell, finding.text, evidence_id)
            ),
            version=version,
            section_id=finding.section_id,
            technology=finding.technology,
            perspective=task.agent,
            criterion=finding.criterion,
            text=finding.text,
            claim_type=finding.claim_type,
            evidence_ids=(evidence_id,),
        )
        cards.append(card)
        claims.append(claim)
    covered = {(c.section_id, c.technology, c.criterion) for c in claims}
    for section, technology, criterion in request.cells():
        if (section, technology, criterion) not in covered:
            missing.append(
                f"{section}/{technology}/{criterion}: original evidence is insufficient"
            )
    for gap in extraction.gaps:
        if (gap.section_id, gap.technology, gap.criterion) in cells:
            missing.append(
                f"{gap.section_id}/{gap.technology}/{gap.criterion}: {gap.reason}"
            )
    drafts = []
    all_limitations = tuple(dict.fromkeys((*limitations, *extraction.limitations)))
    for section_id in task.section_ids:
        section_claims = tuple(c for c in claims if c.section_id == section_id)
        section_missing = tuple(m for m in missing if m.startswith(section_id + "/"))
        drafts.append(
            SectionDraft(
                section_id=section_id,
                plan_version=task.plan_version,
                version=task.draft_versions.get(section_id, 1),
                owner=task.agent,
                claims=section_claims,
                limitations=tuple(dict.fromkeys((*section_missing, *all_limitations))),
                status="unavailable"
                if not section_claims
                else ("needs_revision" if section_missing else "draft"),
            )
        )
    status = "failed" if failed else "insufficient_evidence" if missing else "ok"
    emit(
        "worker_result",
        status=status,
        details={
            "drafts": len(drafts),
            "claims": len(claims),
            "evidence": len(cards),
            "missing_items": missing,
            "usage": usage,
            "next_step": "supervisor_review",
        },
    )
    return TaskResult(
        task_id=task.task_id,
        plan_version=task.plan_version,
        round_id=task.round_id,
        agent=task.agent,
        status=status,
        drafts=tuple(drafts),
        evidence=tuple(cards),
        missing_items=tuple(dict.fromkeys(missing)),
        errors=tuple(dict.fromkeys(errors)),
        usage=usage,
    )


class ResearchWorker:
    """Reusable role configuration; each invocation has its own budget and state."""

    def __init__(self, agent: Worker, *, collector=None, model=None, draft=None):
        self.agent = agent
        self.collector = collector or SourceCollector()
        self.model = model
        self.draft = draft
        with (PROMPTS_DIR / f"{agent}.yaml").open(encoding="utf-8") as stream:
            self.system_prompt = yaml.safe_load(stream)["worker_system_prompt"]

    def run(
        self,
        request: WorkerInput,
        *,
        budget: BudgetLedger | None = None,
        config: RunnableConfig | None = None,
    ) -> TaskResult:
        request = WorkerInput.model_validate(request)
        if request.task.agent != self.agent:
            raise ValueError("Assignment was sent to a different worker")
        counter = TaskBudget(request, budget or BudgetLedger())

        def retrieve_sources(state):
            sources = self.collector.collect(request, counter)
            return {
                "sources": sources,
                "status": "insufficient_evidence"
                if not sources.snapshots
                else ("needs_retry" if sources.errors or sources.limitations else "ok"),
            }

        def draft_sections(state, config: RunnableConfig):
            if not state["sources"].snapshots:
                emit(
                    "model_skipped",
                    status="insufficient_evidence",
                    message="원문이 없어 초안 호출을 생략합니다",
                )
                return {"status": "insufficient_evidence"}
            try:
                counter.reserve(model_calls=1)
                if self.draft:
                    output = self.draft(request, state["sources"], config)
                else:
                    model = self.model or get_llm()
                    output = model.with_structured_output(WorkerExtraction).invoke(
                        [
                            SystemMessage(content=self.system_prompt),
                            HumanMessage(content=_prompt(request, state["sources"])),
                        ],
                        config=config,
                    )
                return {"extraction": WorkerExtraction.model_validate(output)}
            except BudgetExceeded as error:
                return {"limitations": [str(error)], "status": "insufficient_evidence"}
            except Exception as error:  # noqa: BLE001 - external boundary returns task deficits
                return {
                    "failed": True,
                    "status": "failed",
                    "errors": [f"Worker drafting: {type(error).__name__}: {error}"],
                }

        def return_result(state):
            sources = state["sources"]
            output = _assemble(
                request,
                state.get(
                    "extraction", WorkerExtraction(findings=(), gaps=(), limitations=())
                ),
                sources,
                [*sources.errors, *state.get("errors", [])],
                [*sources.limitations, *state.get("limitations", [])],
                state.get("failed", False),
                counter.snapshot(),
            )
            return {"output": output}

        graph = StateGraph(WorkerState)
        for name, function in (
            ("retrieve_sources", retrieve_sources),
            ("draft_sections", draft_sections),
            ("return_result", return_result),
        ):
            add_logged_node(graph, name, function, path=f"workers.{self.agent}.{name}")
        graph.add_edge(START, "retrieve_sources")
        graph.add_edge("retrieve_sources", "draft_sections")
        graph.add_edge("draft_sections", "return_result")
        graph.add_edge("return_result", END)
        return graph.compile(name=f"{self.agent}_worker").invoke(
            {"request": request, "task": request.task.model_dump(mode="json")},
            config=config,
        )["output"]


@lru_cache(maxsize=4)
def get_worker(agent: Worker) -> ResearchWorker:
    return ResearchWorker(agent)


def run_task(
    agent: Worker,
    request: WorkerInput,
    *,
    budget=None,
    config=None,
    worker: ResearchWorker | None = None,
) -> TaskResult:
    return (worker or get_worker(agent)).run(request, budget=budget, config=config)
