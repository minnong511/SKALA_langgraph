"""Behavioral tests for evidence safety and bounded agent execution."""

from __future__ import annotations

import json

from src.agents import market, report, synthesis, technical, verification
from src.agents.base import ResearchOutput, quote_is_present, sanitize_research, source_excerpts
from src.schemas import (
    REPORT_HEADINGS,
    AgentContext,
    AgentRequest,
    AgentResult,
    EvidenceCard,
    ExecutionLimits,
    Finding,
    SourceDocument,
    VerificationVerdict,
)


class FakeLLM:
    def __init__(self, **outputs):
        self.outputs = {key: list(value) for key, value in outputs.items()}
        self.calls = []

    def with_structured_output(self, schema):
        owner = self

        class Bound:
            def invoke(self, messages):
                owner.calls.append((schema.__name__, json.loads(messages[-1][1])))
                output = owner.outputs[schema.__name__].pop(0)
                if isinstance(output, Exception):
                    raise output
                return output

        return Bound()


class Search:
    def __init__(self, sources):
        self.sources, self.calls = sources, []

    def search(self, query, limit=5):
        self.calls.append(query)
        return self.sources[:limit]


class Reader:
    def __init__(self, replacements=None):
        self.calls = []
        self.replacements = replacements or {}

    def read(self, source):
        self.calls.append(source.source_id)
        content = self.replacements.get(source.source_id, source.content)
        if isinstance(content, Exception):
            raise content
        return source.model_copy(update={"content": content})


def request(**kwargs):
    return AgentRequest(run_id="test", task_id="task", limits=ExecutionLimits(max_search_retries=0), **kwargs)


def source():
    return SourceDocument(
        source_id="s1",
        title="Original paper",
        author="A. Author",
        content="The prototype reduced memory use under the evaluated workload.",
        url="https://example.org/paper",
        published_at="2025-01-01",
        accessed_at="2026-09-22",
        page_or_section="p. 2",
    )


def card(evidence_id="technical-e1", **updates):
    values = dict(
        evidence_id=evidence_id,
        technology="TurboQuant",
        perspective="trl",
        claim="The authors evaluate a prototype.",
        source_id="s1",
        evidence_text=source().content,
        source_agent="technical",
        source_url=source().url,
        conditions="evaluated workload",
    )
    values.update(updates)
    return EvidenceCard(**values)


def finding(evidence_ids=None):
    return Finding(
        finding_id="f1",
        technology="TurboQuant",
        perspective="trl",
        statement="A prototype was evaluated.",
        evidence_ids=evidence_ids or ["technical-e1"],
    )


def research_result(cards=None):
    return AgentResult(
        task_id="technical",
        agent="technical",
        status="completed",
        summary="Technical evidence",
        evidence_cards=cards or [card()],
        findings=[finding()],
        sources=[source()],
    )


def reviewed_context(status="verified", **kwargs):
    result = research_result()
    reviewed = AgentResult(
        task_id="verification",
        agent="verification",
        status="completed",
        summary="Reviewed",
        verification=[VerificationVerdict(evidence_id="technical-e1", status=status, reason="Checked")],
    )
    return AgentContext(results={"technical": result, "verification": reviewed}, **kwargs)


def research_output(cards=None, findings=None, missing=None):
    return dict(
        summary="Research",
        evidence_cards=[item.model_dump() for item in (cards or [card()])],
        findings=[item.model_dump() for item in (findings or [finding()])],
        missing_information=missing or [],
    )


def synthesis_output(ids=None):
    ids = ["technical-e1"] if ids is None else ids
    return dict(
        comparison_matrix=[
            dict(
                perspective=perspective,
                turboquant="Conditional assessment",
                itme="Evidence remains limited",
                evidence_ids=ids,
                confidence="high",
            )
            for perspective in ("trl", "market", "stakeholder", "domain")
        ],
        agreements=[],
        conflicts=[],
        conditional_findings=[],
        neutral_conclusion="Conditional conclusions",
        conclusion_evidence_ids=ids,
        neutrality_check="No winner selected",
    )


def report_text(citation="technical-e1"):
    return "\n\n".join(
        "## "
        + heading
        + "\n\n"
        + (
            f"A supported assessment [{citation}]."
            if heading != "REFERENCE"
            else "Invented bibliography to be replaced."
        )
        for heading in REPORT_HEADINGS
    )


def test_quote_check_allows_line_wrap_but_never_paraphrase():
    assert quote_is_present("reduced memory", "reduced\n memory")
    assert not quote_is_present("eliminated memory", "reduced memory")
    assert not quote_is_present(" ", "reduced memory")


def test_excerpt_selection_uses_exact_original_and_rejects_foreign_excerpt():
    original = source()
    excerpt = source_excerpts(original)[0]
    proposed = card(evidence_text="Summary ... merged quote", source_excerpt_id=excerpt["excerpt_id"])
    output = ResearchOutput.model_validate(research_output(cards=[proposed]))
    cards, findings, gaps = sanitize_research(output, {"s1": original}, "technical", set())
    assert cards[0].evidence_text == original.content
    assert findings and not gaps
    assert proposed.evidence_text == "Summary ... merged quote"
    foreign = source_excerpts(original.model_copy(update={"source_id": "s2"}))[0]
    output.evidence_cards[0].source_excerpt_id = foreign["excerpt_id"]
    cards, findings, gaps = sanitize_research(output, {"s1": original}, "technical", set())
    assert not cards and not findings
    assert any("발췌 ID" in gap for gap in gaps)


def test_pdf_page_duplicates_are_read_once_and_quotes_audited():
    first = source().model_copy(
        update={
            "file_path": "paper.pdf",
            "metadata": {
                "file_sha256": "fingerprint",
                "page": 1,
            },
        }
    )
    second = first.model_copy(update={"source_id": "s2"})
    llm = FakeLLM(SearchQueryPlan=[{"queries": ["one", "two"]}], ResearchOutput=[research_output()])
    reader = Reader()
    result = technical.run(
        request(), AgentContext(llm=llm, retriever=Search([first, second]), source_reader=reader)
    )
    assert reader.calls == ["s1"]
    assert len(result.sources) == 1
    payload = llm.calls[-1][1]
    assert "content" not in payload["sources"][0]
    assert payload["sources"][0]["excerpts"][0]["text"] == first.content
    assert result.data["citation_audit"][0]["cards"][0]["category"] == "exact_quote"


def test_research_uses_actual_sources_and_feeds_context_into_both_prompts():
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["query"]}],
        ResearchOutput=[
            research_output(
                cards=[card("market-e1", perspective="market")], findings=[finding(["market-e1"])]
            )
        ],
    )
    ctx = AgentContext(
        llm=llm,
        web_search=Search([source()]),
        source_reader=Reader(),
        results={"technical": research_result()},
    )
    result = market.run(request(feedback=["Check baseline"]), ctx)
    assert result.status == "completed"
    assert result.evidence_cards[0].source_title == "Original paper"
    assert result.evidence_cards[0].source_agent == "market"
    for _, payload in llm.calls:
        assert payload["feedback"] == ["Check baseline"]
        assert payload["technical_result"]["summary"] == "Technical evidence"


def test_research_rejects_invented_source_and_drops_dependent_finding():
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["query"]}],
        ResearchOutput=[research_output(cards=[card(source_id="invented")])],
    )
    result = technical.run(
        request(), AgentContext(llm=llm, retriever=Search([source()]), source_reader=Reader())
    )
    assert result.status == "failed"
    assert result.evidence_cards == []
    assert result.findings == []
    assert any("없는 출처" in gap for gap in result.gaps)


def test_research_repairs_invented_quote_without_another_search():
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["first"]}, {"queries": ["second"]}],
        ResearchOutput=[research_output(cards=[card(evidence_text="Fabricated quote")]), research_output()],
    )
    req = request().model_copy(update={"limits": ExecutionLimits(max_search_retries=1)})
    search = Search([source()])
    result = technical.run(req, AgentContext(llm=llm, retriever=search, source_reader=Reader()))
    assert result.status == "completed"
    assert len(search.calls) == 1
    assert llm.calls[2][1]["citation_errors"]
    assert result.data["citation_audit"][-1]["repair_only"]


def test_research_search_empty_retries_bounded_and_returns_failure():
    llm = FakeLLM(SearchQueryPlan=[{"queries": ["q"]} for _ in range(3)])
    search = Search([])
    req = request().model_copy(update={"limits": ExecutionLimits(max_search_retries=2)})
    result = technical.run(req, AgentContext(llm=llm, retriever=search, source_reader=Reader()))
    assert result.status == "failed"
    assert len(search.calls) == 2  # No new sources: stop instead of repeating a third search.


def test_verification_reopens_original_and_overrides_fabricated_quote_approval():
    llm = FakeLLM(
        VerificationOutput=[
            {
                "summary": "ok",
                "verification": [
                    {"evidence_id": "technical-e1", "status": "verified", "reason": "model says yes"}
                ],
            }
        ]
    )
    reader = Reader({"s1": "The original says something different."})
    result = verification.run(
        request(), AgentContext(llm=llm, source_reader=reader, results={"technical": research_result()})
    )
    assert reader.calls == ["s1"]
    assert result.verification[0].status == "rejected"
    assert result.follow_up_requests[0].target_agent == "technical"
    assert not result.data["passed"]


def test_verification_missing_verdict_fails_closed():
    llm = FakeLLM(VerificationOutput=[{"summary": "ok", "verification": []}])
    result = verification.run(
        request(), AgentContext(llm=llm, source_reader=Reader(), results={"technical": research_result()})
    )
    assert result.verification[0].status == "rejected"


def test_verification_downgrades_performance_claim_without_conditions():
    evidence = card(claim="It saves 70% memory.", conditions="")
    llm = FakeLLM(
        VerificationOutput=[
            {
                "summary": "ok",
                "verification": [
                    {"evidence_id": "technical-e1", "status": "verified", "reason": "quote matches"}
                ],
            }
        ]
    )
    result = verification.run(
        request(),
        AgentContext(llm=llm, source_reader=Reader(), results={"technical": research_result([evidence])}),
    )
    assert result.verification[0].status == "uncertain"


def test_synthesis_cannot_use_unreviewed_evidence():
    ctx = AgentContext(results={"technical": research_result()})
    assert synthesis.run(request(), ctx).status == "failed"
    ctx.results["verification"] = AgentResult(
        task_id="v", agent="verification", status="completed", summary="empty"
    )
    assert synthesis.run(request(), ctx).status == "failed"


def test_synthesis_drops_whole_claim_with_unknown_evidence_id():
    ctx = reviewed_context(llm=FakeLLM(SynthesisOutput=[synthesis_output(["technical-e1", "invented"])]))
    result = synthesis.run(request(), ctx)
    assert result.status == "failed"
    assert result.data["comparison_matrix"] == []
    assert result.used_evidence_ids == []


def test_synthesis_marks_uncertainty_and_keeps_all_four_perspectives():
    ctx = reviewed_context("uncertain", llm=FakeLLM(SynthesisOutput=[synthesis_output()]))
    result = synthesis.run(request(), ctx)
    assert len(result.data["comparison_matrix"]) == 4
    assert all(
        item["confidence"] == "low" and "불확실" in item["conflict_or_condition"]
        for item in result.data["comparison_matrix"]
    )
    assert "불확실" in result.summary


def test_report_rebuilds_references_and_marks_uncertain_citations():
    ctx = reviewed_context("uncertain", llm=FakeLLM(ReportOutput=[{"markdown": report_text()}]))
    ctx.results["synthesis"] = AgentResult(
        task_id="s",
        agent="synthesis",
        status="completed",
        summary="Synthesis",
        used_evidence_ids=["technical-e1"],
        data=synthesis_output(),
    )
    result = report.run(request(), ctx)
    assert result.report_markdown
    assert result.used_evidence_ids == ["technical-e1"]
    assert "Invented bibliography" not in result.report_markdown
    assert "Original paper" in result.report_markdown
    assert "(불확실한 근거) [technical-e1]" in result.report_markdown
    assert len(result.data["reference_entries"]) == 1


def test_report_fails_when_unknown_citation_or_wrong_structure():
    for text in (report_text("invented"), report_text().replace("## 2. 기술 선정", "## Selected technology")):
        ctx = reviewed_context(llm=FakeLLM(ReportOutput=[{"markdown": text}]))
        ctx.results["synthesis"] = AgentResult(
            task_id="s", agent="synthesis", status="completed", summary="Synthesis"
        )
        result = report.run(request(), ctx)
        assert result.status == "failed"
        assert result.report_markdown == text
        assert result.data["draft_markdown"]


def test_verification_rejection_propagates_through_reverse_order_inference_chain():
    cards = [
        card("third", supporting_evidence_ids=["second"], statement_type="analysis_inference"),
        card("second", supporting_evidence_ids=["first"], statement_type="analysis_inference"),
        card("first", evidence_text="Fabricated quote"),
    ]
    llm = FakeLLM(
        VerificationOutput=[
            {
                "summary": "Reviewed",
                "verification": [
                    {"evidence_id": item.evidence_id, "status": "verified", "reason": "Model approval"}
                    for item in cards
                ],
            }
        ]
    )
    result = verification.run(
        request(),
        AgentContext(llm=llm, source_reader=Reader(), results={"technical": research_result(cards)}),
    )
    assert {item.status for item in result.verification} == {"rejected"}


def test_verification_rejects_future_rfc_publication_date():
    result = research_result()
    result.sources[0].published_at = "Fri, 01 Jan 2027 12:00:00 GMT"
    llm = FakeLLM(
        VerificationOutput=[
            {
                "summary": "Reviewed",
                "verification": [
                    {"evidence_id": "technical-e1", "status": "verified", "reason": "Model approval"}
                ],
            }
        ]
    )
    verdict = verification.run(
        request(as_of_date="2026-09-22"),
        AgentContext(llm=llm, source_reader=Reader(), results={"technical": result}),
    )
    assert verdict.verification[0].status == "rejected"
    assert "기준일 이후" in verdict.verification[0].reason


def test_verification_accepts_equivalent_tracking_urls_but_rejects_invented_locator():
    llm = FakeLLM(
        VerificationOutput=[
            {
                "summary": "Reviewed",
                "verification": [
                    {"evidence_id": "technical-e1", "status": "verified", "reason": "Model approval"}
                ],
            }
        ]
        * 2
    )
    result = research_result([card(source_url="https://example.org/paper?utm_source=other#abstract")])
    ctx = AgentContext(llm=llm, source_reader=Reader(), results={"technical": result})
    assert verification.run(request(), ctx).verification[0].status == "verified"
    result.evidence_cards[0].page_or_section = "p. 999"
    assert verification.run(request(), ctx).verification[0].status == "rejected"


def test_ordinary_research_limitations_are_disclosed_without_forcing_retry():
    output = research_output()
    output["limitations"] = ["Public information only"]
    llm = FakeLLM(SearchQueryPlan=[{"queries": ["query"]}], ResearchOutput=[output])
    result = technical.run(
        request(), AgentContext(llm=llm, retriever=Search([source()]), source_reader=Reader())
    )
    assert result.status == "completed"
    assert result.data["limitations"] == ["Public information only"]
    assert result.follow_up_requests == []


def test_research_clears_unverified_locator_in_source_metadata():
    actual = source().model_copy(update={"page_or_section": ""})
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["query"]}],
        ResearchOutput=[research_output(cards=[card(page_or_section="p. 999")])],
    )
    result = technical.run(
        request(), AgentContext(llm=llm, retriever=Search([actual]), source_reader=Reader())
    )
    assert result.evidence_cards[0].page_or_section == ""
    assert "page_or_section" in result.evidence_cards[0].missing_metadata


def test_research_drops_all_dependents_of_invalid_inference():
    cards = [
        card("third", supporting_evidence_ids=["second"]),
        card("second", supporting_evidence_ids=["first"]),
        card("first", evidence_text="Fabricated quote"),
    ]
    llm = FakeLLM(
        SearchQueryPlan=[{"queries": ["query"]}],
        ResearchOutput=[research_output(cards=cards, findings=[finding(["third"])])],
    )
    result = technical.run(
        request(), AgentContext(llm=llm, retriever=Search([source()]), source_reader=Reader())
    )
    assert result.status == "failed"
    assert result.evidence_cards == []
    assert result.findings == []


def test_synthesis_requests_research_only_when_missing_perspective_has_no_evidence():
    output = synthesis_output()
    output["comparison_matrix"] = [
        item for item in output["comparison_matrix"] if item["perspective"] != "market"
    ]
    ctx = reviewed_context(llm=FakeLLM(SynthesisOutput=[output]))
    missing = synthesis.run(request(), ctx)
    assert [item.target_agent for item in missing.follow_up_requests] == ["market"]

    ctx = reviewed_context(llm=FakeLLM(SynthesisOutput=[output]))
    ctx.results["market"] = AgentResult(
        task_id="m",
        agent="market",
        status="completed",
        summary="Market evidence",
        evidence_cards=[card("market-e1", perspective="market", source_agent="market")],
        sources=[source()],
    )
    ctx.results["verification"].verification.append(
        VerificationVerdict(evidence_id="market-e1", status="verified", reason="Reviewed")
    )
    omitted_row = synthesis.run(request(), ctx)
    assert [item.target_agent for item in omitted_row.follow_up_requests] == ["synthesis"]


def test_progress_distinguishes_source_materials_from_retrieved_chunks():
    from src.agents.base import source_material_count

    chunks = [
        source().model_copy(
            update={"source_id": f"pdf-{page}", "file_path": "paper.pdf", "page_or_section": str(page)}
        )
        for page in (1, 2, 3)
    ]
    chunks.extend(
        [
            source(),
            source().model_copy(update={"source_id": "tracked", "url": source().url + "?utm_source=test"}),
        ]
    )
    assert len(chunks) == 5
    assert source_material_count(chunks) == 2
