from pathlib import Path
from unittest.mock import Mock

import pytest

from kv_cache_agent.schemas.evidence import EvidenceCard, SourceRef, SourceSnapshot
from kv_cache_agent.schemas.report import DraftClaim
from kv_cache_agent.verification.pipeline import (
    ClaimJudgement,
    JudgementBatch,
    VerificationPipeline,
)
from kv_cache_agent.verification.sources import (
    SourceLoader,
    classify_source,
    parse_pages,
    read_pdf_pages,
    reference_key,
)


def inputs(
    *,
    text="Memory usage is reduced.",
    source_url="https://docs.nvidia.com/page",
    claim_type="fact",
):
    ref = SourceRef(source_id="source", location=source_url, source_type="official")
    evidence = EvidenceCard(
        evidence_id="e",
        technology="TurboQuant",
        perspective="technical",
        criterion="memory",
        claim="Memory usage is reduced.",
        evidence_text="Memory usage is reduced.",
        source_refs=(ref,),
        claim_type="fact",
    )
    claim = DraftClaim(
        claim_id="actual-draft",
        section_id="technical",
        technology="TurboQuant",
        perspective="technical",
        criterion="memory",
        text=text,
        claim_type=claim_type,
        evidence_ids=("e",),
    )
    return ref, evidence, claim


def supporting_judge(claims, _evidence, _sources):
    return JudgementBatch(
        decisions=[
            ClaimJudgement(
                claim_id=c.claim_id,
                support_level="full",
                matched_text="Memory usage is reduced.",
                rationale="Original text directly supports this claim.",
                claim_type_assessment="correct",
                criterion_assessment="relevant",
            )
            for c in claims
        ]
    )


def loader(ref):
    return SourceSnapshot(
        reference=ref,
        acquisition="tavily_extract",
        status="ok",
        content="Memory usage is reduced.",
    )


def test_actual_text_is_judged_and_changed_claim_is_not_covered_by_card_verification():
    _, evidence, claim = inputs(
        text="Memory usage always disappears in every workload."
    )
    seen = []

    def judge(claims, evidence, sources):
        seen.extend(c.text for c in claims)
        return JudgementBatch(
            decisions=[
                ClaimJudgement(
                    claim_id=claims[0].claim_id,
                    support_level="none",
                    rationale="Reduction does not establish elimination.",
                    claim_type_assessment="unclear",
                )
            ]
        )

    result = VerificationPipeline(loader=loader, judge=judge).verify(
        [claim], [evidence]
    )
    assert seen == [claim.text]
    assert result.decisions[0].status == "unsupported"
    assert result.revision_requests[0]["claim_id"] == "actual-draft"


def test_fabricated_numeric_inference_is_rejected_before_model_call():
    _, evidence, claim = inputs(text="Cost fell by 987654321%.", claim_type="inference")
    judge = Mock(side_effect=supporting_judge)
    result = VerificationPipeline(loader=loader, judge=judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "unsupported"
    assert any("Numeric" in item for item in result.decisions[0].issues)
    judge.assert_not_called()


def test_partial_search_excerpt_and_unknown_host_never_become_verified_facts():
    _, evidence, claim = inputs(source_url="https://example.org/blogless")
    result = VerificationPipeline(loader=loader, judge=supporting_judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "partially_verified"
    assert result.coverage[0].status == "unsupported"
    _, evidence, claim = inputs()

    def excerpt(ref):
        return loader(ref).model_copy(update={"acquisition": "search_excerpt"})

    result = VerificationPipeline(loader=excerpt, judge=supporting_judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "partially_verified"


def test_quote_and_judge_id_checks_prevent_false_promotion():
    _, evidence, claim = inputs()

    def fabricated_quote(claims, *_):
        return JudgementBatch(
            decisions=[
                ClaimJudgement(
                    claim_id=claims[0].claim_id,
                    support_level="full",
                    matched_text="Invented quote",
                    rationale="claimed match",
                    claim_type_assessment="correct",
                    criterion_assessment="relevant",
                )
            ]
        )

    result = VerificationPipeline(loader=loader, judge=fabricated_quote).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "unsupported"
    result = VerificationPipeline(
        loader=loader, judge=lambda *_: JudgementBatch()
    ).verify([claim], [evidence])
    assert result.status == "failed"
    assert result.decisions[0].status == "unsupported"


def test_verified_decision_is_reused_only_for_same_claim_source_and_policy():
    ref, evidence, claim = inputs()
    judge = Mock(side_effect=supporting_judge)
    content = ["Memory usage is reduced."]
    source_loader = lambda ref: SourceSnapshot(
        reference=ref, acquisition="tavily_extract", status="ok", content=content[0]
    )
    service = VerificationPipeline(loader=source_loader, judge=judge)
    first = service.verify([claim], [evidence])
    service.verify([claim], [evidence])
    assert judge.call_count == 1
    snapshots = {reference_key(ref): source_loader(ref)}
    assert first.decisions[0].applies_to(claim, snapshots)
    content[0] += " Updated condition."
    service.verify([claim], [evidence])
    assert judge.call_count == 2
    assert not first.decisions[0].applies_to(
        claim, {reference_key(ref): source_loader(ref)}
    )
    service.verify([claim.model_copy(update={"version": 2})], [evidence])
    assert judge.call_count == 3


def test_two_pdf_pages_and_non_monotonic_locators_are_not_conflated():
    root = Path(__file__).parents[1]
    paper = root / "data/papers/turboquant.pdf"
    read = SourceLoader(root=root)
    page2 = read.load(SourceRef(source_id="paper", location=str(paper), pages=(2,)))
    page17 = read.load(SourceRef(source_id="paper", location=str(paper), pages=(17,)))
    assert page2.content != page17.content
    mixed = read.load(
        SourceRef(source_id="paper", location=str(paper), locator="p. 20; p. 2")
    )
    assert page2.content in mixed.content
    assert read_pdf_pages(paper, pages=(20,)) in mixed.content
    assert parse_pages("pp. 2-4; p. 1") == (2, 3, 4, 1)


def test_multiple_documents_are_fetched_and_linked_separately():
    _, evidence, claim = inputs()
    second = SourceRef(
        source_id="cxl-paper", location="data/papers/cxl_based_kv_cache.pdf", pages=(2,)
    )
    evidence = evidence.model_copy(
        update={"source_refs": (*evidence.source_refs, second)}
    )
    loaded = []

    def load(ref):
        loaded.append(ref.location)
        return loader(ref).model_copy(
            update={"acquisition": "local_pdf" if ref is second else "tavily_extract"}
        )

    result = VerificationPipeline(loader=load, judge=supporting_judge).verify(
        [claim], [evidence]
    )
    assert len(loaded) == 2
    assert len(result.decisions[0].source_versions) == 2


def test_hostname_boundary_does_not_trust_lookalikes():
    assert classify_source("https://docs.nvidia.com/page") == "official"
    assert classify_source("https://notnvidia.com/page") == "unknown"
    assert classify_source("https://nvidia.com.evil.test/page") == "unknown"


def test_loader_cannot_substitute_another_page_of_the_same_document():
    _, evidence, claim = inputs()
    ref = SourceRef(
        source_id="paper", location="data/papers/turboquant.pdf", pages=(2,)
    )
    evidence = evidence.model_copy(update={"source_refs": (ref,)})
    judge = Mock(side_effect=supporting_judge)

    def wrong_page(ref):
        return SourceSnapshot(
            reference=ref.model_copy(update={"pages": (17,)}),
            acquisition="local_pdf",
            status="ok",
            content="Memory usage is reduced.",
        )

    result = VerificationPipeline(loader=wrong_page, judge=judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "unsupported"
    assert result.status == "failed"
    judge.assert_not_called()


def test_unknown_evidence_id_never_becomes_a_supported_claim():
    _, evidence, claim = inputs()
    judge = Mock(side_effect=supporting_judge)
    claim = claim.model_copy(update={"evidence_ids": ("missing",)})
    result = VerificationPipeline(loader=loader, judge=judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status == "unsupported"
    judge.assert_not_called()


def test_model_context_has_original_text_and_links_not_the_authors_fake_quote(
    monkeypatch,
):
    import json

    from kv_cache_agent.verification import pipeline

    ref, evidence, claim = inputs()
    evidence = evidence.model_copy(update={"evidence_text": "AUTHOR-INVENTED-QUOTE"})
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = supporting_judge(
        [claim], {}, {}
    )
    monkeypatch.setattr(pipeline, "get_llm", lambda: llm)
    result = VerificationPipeline(loader=loader).verify([claim], [evidence])
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    context = json.loads(messages[-1].content)
    assert "AUTHOR-INVENTED-QUOTE" not in messages[-1].content
    assert (
        context["original_sources"][reference_key(ref)]["content"]
        == "Memory usage is reduced."
    )
    assert result.decisions[0].status == "verified"


def test_oversized_source_requires_explicit_scoping_without_a_model_call(monkeypatch):
    from kv_cache_agent.verification import pipeline

    _, evidence, claim = inputs()
    llm = Mock()
    monkeypatch.setattr(pipeline, "get_llm", lambda: llm)
    source = lambda ref: SourceSnapshot(
        reference=ref,
        acquisition="tavily_extract",
        status="ok",
        content="Memory usage is reduced. " * 4000,
    )
    result = VerificationPipeline(loader=source).verify([claim], [evidence])
    assert result.status == "failed"
    assert "select versioned source spans" in result.errors[0]
    llm.with_structured_output.assert_not_called()


def test_implicit_truncation_is_partial_and_invalidates_prior_receipt():
    ref, evidence, claim = inputs()
    service = VerificationPipeline(loader=loader, judge=supporting_judge)
    approved = service.verify([claim], [evidence]).decisions[0]
    clipped = loader(ref).model_copy(update={"truncated": True})
    assert not approved.applies_to(claim, {reference_key(ref): clipped})
    result = VerificationPipeline(
        loader=lambda _: clipped, judge=supporting_judge
    ).verify([claim], [evidence])
    assert result.decisions[0].status == "partially_verified"


@pytest.mark.parametrize("assessment", ("irrelevant", "unclear"))
def test_true_source_fact_cannot_fill_an_unanswered_criterion(assessment):
    _ref, evidence, claim = inputs()
    claim = claim.model_copy(update={"criterion": "adoption"})
    evidence = evidence.model_copy(update={"criterion": "adoption"})

    def judge(claims, cards, sources):
        return JudgementBatch(
            decisions=[
                ClaimJudgement(
                    claim_id=c.claim_id,
                    support_level="full",
                    matched_text="Memory usage is reduced.",
                    rationale="Memory reduction does not demonstrate actual adoption.",
                    claim_type_assessment="correct",
                    criterion_assessment=assessment,
                )
                for c in claims
            ]
        )

    result = VerificationPipeline(loader=loader, judge=judge).verify(
        [claim], [evidence]
    )
    assert result.decisions[0].status != "verified"
    assert result.coverage[0].status == "unsupported"
    assert result.revision_requests


def test_multi_worker_verification_batches_are_bounded_by_unique_source_size():
    refs = []
    evidence = []
    claims = []
    snapshots = {}
    for index in range(2):
        ref, card, claim = inputs(source_url=f"https://docs.nvidia.com/page{index}")
        ref = ref.model_copy(update={"source_id": f"source-{index}"})
        card = card.model_copy(
            update={"evidence_id": f"e-{index}", "source_refs": (ref,)}
        )
        claim = claim.model_copy(
            update={"claim_id": f"c-{index}", "evidence_ids": (card.evidence_id,)}
        )
        snapshots[ref.source_id] = SourceSnapshot(
            reference=ref,
            content="Memory usage is reduced. " + ("padding " * 8000),
            acquisition="tavily_extract",
            status="ok",
        )
        refs.append(ref)
        evidence.append(card)
        claims.append(claim)
    calls = []

    def judge(batch, cards, sources):
        calls.append(len(batch))
        return supporting_judge(batch, cards, sources)

    result = VerificationPipeline(
        loader=lambda ref: snapshots[ref.source_id], judge=judge
    ).verify(claims, evidence)
    assert result.status == "ok"
    assert calls == [1, 1]
