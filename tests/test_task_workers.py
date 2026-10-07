"""Worker contract, original acquisition, revision and verification integration."""

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import httpx
import pytest
from pydantic import ValidationError
from reportlab.pdfgen.canvas import Canvas

from kv_cache_agent.agents.task_worker import ResearchWorker
from kv_cache_agent.observability.logger import RunSession
from kv_cache_agent.observability.tracing import TracingSettings
from kv_cache_agent.schemas.evidence import ClaimAttribution, SourceRef
from kv_cache_agent.schemas.report import ReportPlan, SectionSpec
from kv_cache_agent.schemas.research import BudgetLedger, BudgetLimits, ResearchTask
from kv_cache_agent.schemas.worker import (
    SourceQuote,
    WorkerExtraction,
    WorkerFinding,
    WorkerInput,
)
from kv_cache_agent.tools.research_sources import SourceCollector
from kv_cache_agent.tools.tavily_extract import SourceCache, TavilyExtractor
from kv_cache_agent.verification.pipeline import (
    ClaimJudgement,
    JudgementBatch,
    VerificationPipeline,
)
from kv_cache_agent.verification.sources import SourceLoader

ROLES = ("technical", "market", "stakeholder", "cloud_domain")
TECHNOLOGIES = ("TurboQuant", "CXL-based")
BODIES = {
    "https://google.com/turboquant": "Google says TurboQuant reduces KV cache memory.",
    "https://cxlconsortium.org/cache": "CXL Consortium says CXL-based expands KV cache memory.",
}


def request(role="technical", **changes):
    plan = ReportPlan(
        plan_id="report",
        user_query="Compare KV cache approaches",
        sections=tuple(
            SectionSpec(
                section_id=role,
                title=role,
                order=i,
                owner=role,
                perspective=role,
                objective="Compare effects",
                technologies=TECHNOLOGIES,
                criteria=("memory", "latency"),
            )
            for i, role in enumerate(ROLES)
        ),
    )
    task = ResearchTask(
        task_id=f"task-{role}",
        plan_id=plan.plan_id,
        plan_version=plan.version,
        round_id=1,
        agent=role,
        section_ids=(role,),
        technologies=TECHNOLOGIES,
        criteria=("memory",),
        objective="Write requested memory evaluation",
        max_search_calls=2,
    )
    return WorkerInput(task=task.model_copy(update=changes), plan=plan)


@pytest.fixture
def sources(tmp_path):
    paper = tmp_path / "paper.pdf"
    canvas = Canvas(str(paper))
    for text in BODIES.values():
        canvas.drawString(30, 700, text)
        canvas.showPage()
    canvas.save()
    chunks = [
        {
            "chunk_id": f"p{page}",
            "content": body,
            "metadata": {
                "source_path": str(paper),
                "page": page,
                "source_title": "Research",
                "source_locator": f"p. {page}",
            },
        }
        for page, body in enumerate(BODIES.values(), start=1)
    ]
    search = Mock(
        return_value=[
            {
                "url": url,
                "title": "Search title",
                "content": "fabricated 98765%",
                "raw_content": "search raw excerpt",
                "published_date": "2026-10-01",
            }
            for url in BODIES
        ]
    )
    client = Mock()
    client.extract.side_effect = lambda **kwargs: {
        "results": [{"url": url, "raw_content": BODIES[url]} for url in kwargs["urls"]]
    }
    extractor = TavilyExtractor(client=client, cache=SourceCache(tmp_path / "cache"))
    collector = SourceCollector(
        root=tmp_path,
        search=search,
        paper_search=Mock(return_value=chunks),
        extractor=extractor,
    )
    return collector, extractor, search, paper


def findings(req, collection, _config, *, technology=None, **changes):
    values = []
    for tech in req.task.technologies:
        if technology and tech != technology:
            continue
        source = next((s for s in collection.snapshots if tech in s.content), None)
        if source is None:
            continue
        attribution = None
        if req.task.agent == "stakeholder":
            attribution = ClaimAttribution(
                actor="Google" if tech == "TurboQuant" else "CXL Consortium",
                actor_group="developer",
                statement_kind="public_statement",
                position="positive",
            )
        values.append(
            WorkerFinding(
                **{
                    "section_id": req.task.agent,
                    "technology": tech,
                    "criterion": "memory",
                    "text": source.content,
                    "claim_type": "fact",
                    "confidence": 0.9,
                    "caveat": "",
                    "attribution": attribution,
                    "source_quotes": (
                        SourceQuote(
                            source_id=source.reference.source_id, quote=source.content
                        ),
                    ),
                    **changes,
                }
            )
        )
    return WorkerExtraction(findings=tuple(values), gaps=(), limitations=())


@pytest.mark.parametrize("role", ROLES)
def test_four_workers_return_draft_evidence_and_per_task_usage(role, sources, tmp_path):
    collector, _, search, _ = sources
    worker = ResearchWorker(role, collector=collector, draft=findings)
    ledger = BudgetLedger()
    with RunSession(
        tmp_path / "runs", tracing=TracingSettings(enabled=False), console=False
    ) as run:
        result = worker.run(request(role), budget=ledger)
    assert result.status == "ok"
    assert len(result.drafts[0].claims) == len(result.evidence) == 2
    assert result.drafts[0].status == "draft"
    assert {c.technology for c in result.evidence} == set(TECHNOLOGIES)
    assert all(
        c.perspective == role and c.criterion == "memory" for c in result.evidence
    )
    assert result.usage["model_calls"] == 1
    assert result.usage["search_calls"] == (0 if role == "technical" else 2)
    assert result.usage["extract_calls"] == (0 if role == "technical" else 1)
    assert ledger.snapshot()["model_calls"] == 1
    assert search.call_count == (0 if role == "technical" else 2)
    events = [
        json.loads(line)
        for line in (run.directory / "events.jsonl").read_text().splitlines()
    ]
    starts = [e for e in events if e["event"] == "node_start"]
    assert [e["node_path"].split(".")[-1] for e in starts] == [
        "retrieve_sources",
        "draft_sections",
        "return_result",
    ]
    assert all(
        e["task_id"] == f"task-{role}" and e["criteria"] == ["memory"] for e in starts
    )


def test_technical_draft_is_passed_to_shared_verification(sources):
    collector, _, _, _ = sources
    result = ResearchWorker("technical", collector=collector, draft=findings).run(
        request()
    )

    def judge(claims, _evidence, snapshots):
        return JudgementBatch(
            decisions=[
                ClaimJudgement(
                    claim_id=c.claim_id,
                    support_level="full",
                    matched_text=c.text,
                    rationale="Exact original sentence",
                    claim_type_assessment="correct",
                    criterion_assessment="relevant",
                )
                for c in claims
            ]
        )

    verification = VerificationPipeline(
        loader=SourceLoader(root=collector.root).load, judge=judge
    ).verify(list(result.drafts[0].claims), list(result.evidence))
    assert verification.status == "ok"
    assert all(d.status == "verified" for d in verification.decisions)
    assert len(verification.coverage) == 2


def test_missing_technology_is_returned_without_research_loop(sources):
    collector, _, search, _ = sources
    draft = lambda req, collection, config: findings(
        req, collection, config, technology="TurboQuant"
    )
    result = ResearchWorker("market", collector=collector, draft=draft).run(
        request("market")
    )
    assert result.status == "insufficient_evidence"
    assert any("market/CXL-based/memory" in gap for gap in result.missing_items)
    assert result.drafts[0].status == "needs_revision"
    assert search.call_count == 2


@pytest.mark.parametrize("defect", ("index", "source_link", "stale_chunk"))
def test_technical_missing_index_or_source_returns_deficit(defect, sources):
    collector, _, _, _ = sources
    if defect == "index":
        collector.paper_search.side_effect = FileNotFoundError("No FAISS index")
    else:
        chunks = collector.paper_search.return_value
        for chunk in chunks:
            if defect == "source_link":
                chunk["metadata"]["source_path"] = str(collector.root / "missing.pdf")
            else:
                chunk["content"] = "Stale chunk no longer in PDF"
    draft = Mock()
    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert result.status == "insufficient_evidence"
    assert len(result.missing_items) == 2
    assert result.errors and result.drafts[0].status == "unavailable"
    draft.assert_not_called()
    assert result.usage["model_calls"] == 0


@pytest.mark.parametrize(
    "changes",
    (
        {"section_id": "market"},
        {"criterion": "unassigned"},
        {"source_quotes": (SourceQuote(source_id="invented", quote="claim"),)},
        {"claim_type": "inference", "caveat": ""},
    ),
)
def test_unassigned_claim_or_broken_source_is_rejected(changes, sources):
    collector, _, _, _ = sources
    draft = lambda req, collection, config: findings(req, collection, config, **changes)
    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert not result.evidence and result.status == "insufficient_evidence"


def test_multi_paper_finding_retains_all_document_and_page_refs(sources, tmp_path):
    collector, _, _, paper = sources
    other = tmp_path / "other.pdf"
    other.write_bytes(paper.read_bytes())
    extra = dict(collector.paper_search.return_value[0])
    extra["chunk_id"] = "other-p1"
    extra["metadata"] = {**extra["metadata"], "source_path": str(other)}
    collector.paper_search.return_value.append(extra)

    def draft(req, collection, config):
        result = findings(req, collection, config)
        first = result.findings[0]
        quotes = tuple(
            SourceQuote(source_id=s.reference.source_id, quote=s.content)
            for s in collection.snapshots
            if "TurboQuant" in s.content
        )
        return result.model_copy(
            update={
                "findings": (
                    first.model_copy(update={"source_quotes": quotes}),
                    *result.findings[1:],
                )
            }
        )

    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    refs = result.evidence[0].source_refs
    assert {r.location for r in refs} == {str(paper), str(other)}
    assert all(r.pages == (1,) and r.chunk_ids and r.version for r in refs)


def test_search_snippet_does_not_replace_original_and_yaml_is_used(sources):
    collector, _, _, _ = sources
    model = Mock()

    def invoke(messages, **kwargs):
        payload = json.loads(messages[1].content)
        assert "시장" in messages[0].content
        assert "98765" not in messages[1].content
        assert "search raw excerpt" not in messages[1].content
        assert all("passages" in source for source in payload["original_sources"])
        source = payload["original_sources"][1]
        return WorkerExtraction(
            findings=(
                WorkerFinding(
                    section_id="market",
                    technology="CXL-based",
                    criterion="memory",
                    text=source["passages"][0]["text"],
                    source_quotes=(
                        SourceQuote(
                            source_id=source["source_id"],
                            passage_id=source["passages"][0]["passage_id"],
                        ),
                    ),
                    claim_type="fact",
                    confidence=0.8,
                    caveat="",
                    attribution=None,
                ),
            ),
            gaps=(),
            limitations=(),
        )

    model.with_structured_output.return_value.invoke.side_effect = invoke
    result = ResearchWorker("market", collector=collector, model=model).run(
        request("market", questions=("TurboQuant adoption",))
    )
    assert result.evidence[0].technology == "CXL-based"
    assert len(result.evidence) == 1
    assert any("TurboQuant" in gap for gap in result.missing_items)


def test_unknown_host_cannot_be_promoted_to_official(sources):
    collector, extractor, search, _ = sources
    search.return_value = [
        {"url": "https://unknown.example/cache", "title": "Official"}
    ]
    extractor.client.extract.side_effect = lambda **_: {
        "results": [
            {
                "url": "https://unknown.example/cache",
                "raw_content": next(iter(BODIES.values())),
            }
        ]
    }
    result = ResearchWorker("market", collector=collector, draft=findings).run(
        request("market")
    )
    assert result.evidence[0].source_refs[0].source_type == "unknown"


def test_original_failure_is_not_filled_by_search_excerpt(sources):
    collector, extractor, _, _ = sources
    extractor.client.extract.side_effect = lambda **_: {
        "failed_results": [{"url": url, "error": "403 blocked"} for url in BODIES]
    }
    result = ResearchWorker("market", collector=collector, draft=findings).run(
        request("market")
    )
    assert not result.evidence
    assert any("Original unavailable" in gap for gap in result.drafts[0].limitations)


@pytest.mark.parametrize("kind", ("missing_actor", "invented_actor", "false_statement"))
def test_stakeholder_separates_public_statements_and_inference(kind, sources):
    collector, _, _, _ = sources
    changes = {"attribution": None}
    if kind == "invented_actor":
        changes = {
            "attribution": ClaimAttribution(
                actor="Amazon",
                actor_group="cloud",
                statement_kind="public_statement",
                position="positive",
            )
        }
    if kind == "false_statement":
        changes = {"claim_type": "inference", "caveat": "Only a prediction"}
    draft = lambda req, collection, config: findings(req, collection, config, **changes)
    result = ResearchWorker("stakeholder", collector=collector, draft=draft).run(
        request("stakeholder")
    )
    assert not result.evidence and result.status == "insufficient_evidence"


def test_revision_reuses_sources_without_search_and_advances_version(sources):
    collector, _, search, _ = sources
    worker = ResearchWorker("market", collector=collector, draft=findings)
    first = worker.run(request("market"))
    search.reset_mock()
    base = request(
        "market",
        action="revise",
        round_id=2,
        draft_versions={"market": 2},
    )
    second = worker.run(
        WorkerInput(
            task=base.task.model_copy(
                update={
                    "existing_evidence_ids": tuple(
                        e.evidence_id for e in first.evidence
                    )
                }
            ),
            plan=base.plan,
            existing_evidence=first.evidence,
            existing_drafts=first.drafts,
        )
    )
    search.assert_not_called()
    assert second.status == "ok" and second.usage["search_calls"] == 0
    assert second.usage["extract_calls"] == 0
    assert second.drafts[0].version == 2
    assert all(c.version == 2 for c in second.drafts[0].claims)


def test_cloud_reads_actual_technical_results_preserving_both_pages(sources):
    collector, _, search, paper = sources
    technical = ResearchWorker("technical", collector=collector, draft=findings).run(
        request()
    )
    base = request("cloud_domain", action="revise")
    result = ResearchWorker("cloud_domain", collector=collector, draft=findings).run(
        WorkerInput(task=base.task, plan=base.plan, technical_results=(technical,))
    )
    assert result.status == "ok" and result.usage["search_calls"] == 0
    assert {(r.location, r.pages) for e in result.evidence for r in e.source_refs} == {
        (str(paper), (1,)),
        (str(paper), (2,)),
    }
    search.assert_not_called()
    assert all(e.perspective == "cloud_domain" for e in result.evidence)


def test_model_budget_and_final_reserve_are_preserved(sources):
    collector, _, _, _ = sources
    ledger = BudgetLedger(BudgetLimits(model_calls=1, finish_reserve=1))
    draft = Mock()
    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request(), budget=ledger
    )
    draft.assert_not_called()
    assert result.status == "insufficient_evidence"
    assert ledger.snapshot()["model_calls"] == 0
    ledger.reserve(model_calls=1, finishing=True)


def test_parallel_workers_share_budget_but_keep_individual_usage(sources):
    collector, _, _, _ = sources
    ledger = BudgetLedger(BudgetLimits(model_calls=2, finish_reserve=1))
    worker = ResearchWorker("technical", collector=collector, draft=findings)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda _: worker.run(request(), budget=ledger), range(2))
        )
    assert sum(r.usage["model_calls"] for r in results) == 1
    assert sorted(r.status for r in results) == ["insufficient_evidence", "ok"]


def test_transport_retry_consumes_task_budget_but_missing_evidence_does_not_retry(
    sources,
):
    collector, _, search, _ = sources
    search.side_effect = [httpx.ReadTimeout("temporary"), []]
    result = ResearchWorker("market", collector=collector, draft=findings).run(
        request("market")
    )
    assert result.usage["search_calls"] == search.call_count == 2
    assert result.status == "insufficient_evidence"


def test_invalid_and_stale_worker_assignments_are_rejected():
    base = request()
    with pytest.raises(ValidationError, match="applicable|outside"):
        request(criteria=("unassigned",))
    with pytest.raises(ValidationError, match="Stale"):
        request(plan_version=2)
    with pytest.raises(ValueError, match="different worker"):
        ResearchWorker("market").run(base)


def test_draft_model_failure_returns_failed_task_not_graph_success(sources):
    collector, _, _, _ = sources
    draft = Mock(side_effect=RuntimeError("model failed"))
    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert result.status == "failed" and result.errors
    assert result.usage["model_calls"] == 1


def test_long_web_body_keeps_hash_anchored_spans_and_verifier_can_reload(sources):
    collector, extractor, _, _ = sources
    text = "preface " * 2500 + list(BODIES.values())[1] + " appendix" * 1500
    extractor.client.extract.side_effect = lambda **kwargs: {
        "results": [{"url": url, "raw_content": text} for url in kwargs["urls"]]
    }
    model = Mock()

    def invoke(messages, **kwargs):
        payload = json.loads(messages[1].content)
        source = next(
            s
            for s in payload["original_sources"]
            if any("CXL-based" in p["text"] for p in s["passages"])
        )
        quote = list(BODIES.values())[1]
        return WorkerExtraction(
            findings=(
                WorkerFinding(
                    section_id="market",
                    technology="CXL-based",
                    criterion="memory",
                    text=quote,
                    source_quotes=(
                        SourceQuote(source_id=source["source_id"], quote=quote),
                    ),
                    claim_type="fact",
                    confidence=0.8,
                    caveat="",
                    attribution=None,
                ),
            ),
            gaps=(),
            limitations=(),
        )

    model.with_structured_output.return_value.invoke.side_effect = invoke
    result = ResearchWorker("market", collector=collector, model=model).run(
        request("market")
    )
    ref = result.evidence[0].source_refs[0]
    assert ref.character_range and ref.version
    loaded = SourceLoader(extractor=extractor).load(ref)
    assert list(BODIES.values())[1] in loaded.content
    assert len(loaded.content) <= 6000 and not loaded.truncated
    assert extractor.client.extract.call_count == 1


def test_passage_id_is_resolved_to_exact_source_without_model_rewriting(sources):
    from kv_cache_agent.tools.research_sources import source_passages

    collector, _, _, _ = sources

    def draft(req, collection, config):
        result = findings(req, collection, config)
        revised = []
        for finding in result.findings:
            source = next(
                s for s in collection.snapshots if finding.technology in s.content
            )
            passage = source_passages(source)[0]
            revised.append(
                finding.model_copy(
                    update={
                        "source_quotes": (
                            SourceQuote(
                                source_id=source.reference.source_id,
                                passage_id=passage["passage_id"],
                            ),
                        )
                    }
                )
            )
        return result.model_copy(update={"findings": tuple(revised)})

    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert result.status == "ok"
    assert {e.evidence_text for e in result.evidence} == set(BODIES.values())


def test_unknown_passage_and_rewritten_passage_are_rejected(sources):
    from kv_cache_agent.tools.research_sources import source_passages

    collector, _, _, _ = sources

    def draft(req, collection, config):
        result = findings(req, collection, config)
        revised = []
        for index, finding in enumerate(result.findings):
            source = next(
                s for s in collection.snapshots if finding.technology in s.content
            )
            passage = source_passages(source)[0]
            revised.append(
                finding.model_copy(
                    update={
                        "source_quotes": (
                            SourceQuote(
                                source_id=source.reference.source_id,
                                passage_id="unknown"
                                if index == 0
                                else passage["passage_id"],
                                quote="rewritten" if index else "",
                            ),
                        )
                    }
                )
            )
        return result.model_copy(update={"findings": tuple(revised)})

    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert not result.evidence
    assert any("unknown original passage" in m for m in result.missing_items)
    assert any("rewrote" in m for m in result.missing_items)


def test_cached_pdf_spans_keep_distinct_ranges_and_document_hash(sources):
    collector, _, _, paper = sources
    loader = SourceLoader(root=collector.root)
    base = loader.load(SourceRef(source_id="page", location=str(paper), pages=(1,)))
    first = base.reference.model_copy(
        update={"source_id": "first", "character_range": (0, 10)}
    )
    second = base.reference.model_copy(
        update={"source_id": "second", "character_range": (10, 25)}
    )
    assert loader.load(first).content == base.content[:10]
    assert loader.load(second).content == base.content[10:25]
    assert loader.load(first).reference.version == base.reference.version
    assert loader.load(base.reference).content == base.content


def test_web_version_change_is_rejected_without_silently_rebinding(sources):
    _, extractor, _, _ = sources
    reference = SourceRef(
        source_id="changed", location=next(iter(BODIES)), version="previous-hash"
    )
    with pytest.raises(ValueError, match="Source changed"):
        SourceLoader(extractor=extractor).load(reference)


def test_fabricated_worker_number_is_rejected_by_shared_verifier(sources):
    collector, _, _, _ = sources

    def draft(req, collection, config):
        result = findings(req, collection, config)
        return result.model_copy(
            update={
                "findings": tuple(
                    f.model_copy(update={"text": f.text + " Cost reduced 987654321%."})
                    for f in result.findings
                )
            }
        )

    result = ResearchWorker("technical", collector=collector, draft=draft).run(
        request()
    )
    assert result.status == "ok"  # A draft is not approved evidence.
    judge = lambda claims, evidence, snapshots: JudgementBatch(
        decisions=[
            ClaimJudgement(
                claim_id=c.claim_id,
                support_level="full",
                matched_text=next(iter(snapshots.values())).content,
                rationale="semantic judgement alone cannot approve a fabricated number",
                claim_type_assessment="correct",
                criterion_assessment="relevant",
            )
            for c in claims
        ]
    )
    verification = VerificationPipeline(
        loader=SourceLoader(root=collector.root).load, judge=judge
    ).verify(list(result.drafts[0].claims), list(result.evidence))
    assert all(d.status == "unsupported" for d in verification.decisions)


def test_real_worker_subgraphs_and_shared_verifier_run_inside_supervised_loop(sources):
    from kv_cache_agent.agents.supervisor_control import SupervisorController
    from kv_cache_agent.graph.workflow import build_supervised_workflow
    from kv_cache_agent.schemas.report import SectionSpec
    from kv_cache_agent.schemas.supervision import ReportOutline
    from tests.supervised_fixtures import choose_missing

    collector, extractor, search, _ = sources
    base = request().plan
    sections = tuple(
        s.model_copy(
            update={
                "criteria": ("memory",),
                "questions": ("What original evidence documents memory effects?",),
            }
        )
        for s in base.sections
    )
    sections = (
        *sections,
        SectionSpec(
            section_id="conclusion",
            title="Conclusion",
            order=4,
            owner="supervisor",
            objective="Compare supported results",
        ),
    )
    supervisor = SupervisorController(
        planner=lambda q, config=None: ReportOutline(sections=sections),
        router=choose_missing,
    )
    workers = {
        role: ResearchWorker(role, collector=collector, draft=findings)
        for role in ROLES
    }

    def judge(claims, evidence, snapshots):
        return JudgementBatch(
            decisions=[
                ClaimJudgement(
                    claim_id=c.claim_id,
                    support_level="full",
                    matched_text=c.text,
                    rationale="Exact retrieved original sentence",
                    claim_type_assessment="correct",
                    criterion_assessment="relevant",
                )
                for c in claims
            ]
        )

    result = build_supervised_workflow(
        supervisor=supervisor,
        workers=workers,
        verifier_factory=lambda budget: VerificationPipeline(
            loader=SourceLoader(
                root=collector.root, extractor=extractor, budget=budget
            ).load,
            judge=judge,
            budget=budget,
        ),
    ).invoke({"user_query": base.user_query})
    assert result["status"] == "ready_for_finalization", result.get("fatal_errors")
    assert len(result["task_results"]) == 4 and all(
        c.status == "verified" for c in result["coverage"]
    )
    assert result["budget_usage"]["model_calls"] == 10
    assert search.call_count == result["budget_usage"]["search_calls"] == 6
    assert (
        result["budget_usage"]["extract_calls"] == 1
        and result["budget_usage"]["extract_urls"] == 2
    )
