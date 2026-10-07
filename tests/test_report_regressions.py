"""실제 실행에서 발견한 출처·검색·자동 채움 문제의 회귀 테스트."""

from copy import deepcopy

import httpx
import pytest

from kv_cache_agent.agents import market, stakeholder, verifier
from kv_cache_agent.agents.report_writer import (
    Paragraph,
    _bounded_draft_model,
    _render_markdown,
)
from kv_cache_agent.tools.source_fetcher import fetch_source
from kv_cache_agent.tools.source_metadata import normalize_source


def test_old_computer_path_is_rebased_and_has_verified_bibliography():
    record = {
        "source_url": "/Users/another-user/repo/data/papers/cxl_based_kv_cache.pdf"
    }
    original = deepcopy(record)
    output = normalize_source(record)
    assert record == original
    assert output["source_url"] == "data/papers/cxl_based_kv_cache.pdf"
    assert output["published_date"] == "2026-06-16"
    assert output["canonical_url"].endswith("2606.12556v2")
    assert verifier._fetch_original_source(record)["fetch_status"] == "ok"


@pytest.mark.parametrize("agent", [market, stakeholder])
def test_query_alone_cannot_make_unrelated_results_relevant(agent):
    result = {
        "title": "IHSS service plan",
        "url": "https://example.org/ihss",
        "content": "Cloud provider disability service planning customer guide",
    }
    assert agent._normalize_result(result, "TurboQuant CXL KV cache adoption") is None


def test_publication_metadata_is_extracted_before_removing_scripts():
    response = httpx.Response(
        200,
        request=httpx.Request("GET", "https://example.org/a"),
        headers={"content-type": "text/html"},
        text="""
        <html><head><title>Article</title>
        <meta property="article:published_time" content="2026-09-15">
        <script type="application/ld+json">{"author":{"name":"Research Team"}}</script>
        </head><body><main>KV cache research</main></body></html>""",
    )

    class Client:
        def get(self, url):
            return response

    output = fetch_source("https://example.org/a", client=Client())
    assert output["published_date"] == "2026-09-15"
    assert output["authors"] == "Research Team"
    assert "Research Team" not in output["content"]


def test_reference_numbers_are_per_source_not_per_card():
    card = normalize_source(
        {"source_url": "data/papers/turboquant.pdf", "source_locator": "p. 1"}
    )
    cards = {"a": card, "b": {**card, "source_locator": "p. 2"}}
    config = {
        "report_structure": [
            {"id": "summary", "title": "SUMMARY"},
            {"id": "reference", "title": "REFERENCE"},
        ]
    }
    text = _render_markdown(
        config,
        {
            "summary": [
                Paragraph(text="원문 설명", claim_type="fact", evidence_ids=["a", "b"])
            ]
        },
        cards,
    )
    assert text.count("[1]") == 2
    assert "[2]" not in text
    assert "2504.19874v1" in text and "2025-04-28" in text
    assert "p. 1; p. 2" in text


def test_generated_schema_rejects_made_up_evidence_id():
    model = _bounded_draft_model(
        {"known": {"verification_status": "verified"}}, ["summary"]
    )
    with pytest.raises(ValueError):
        model.model_validate(
            {
                "summary": [
                    {"text": "설명", "claim_type": "fact", "evidence_ids": ["made-up"]}
                ]
            }
        )


def test_generated_schema_restricts_partial_evidence_to_nonfacts():
    model = _bounded_draft_model(
        {
            "verified": {"verification_status": "verified"},
            "partial": {"verification_status": "partially_verified"},
        },
        ["summary"],
    )
    paragraph = {"text": "설명", "claim_type": "fact", "evidence_ids": ["partial"]}
    with pytest.raises(ValueError):
        model.model_validate({"summary": [paragraph]})
    paragraph["claim_type"] = "inference"
    assert model.model_validate({"summary": [paragraph]})


def test_every_chapter_section_is_a_required_field():
    model = _bounded_draft_model({}, ["section_1_1", "section_1_2"])
    paragraph = {"text": "자료 한계", "claim_type": "limitation", "evidence_ids": []}
    with pytest.raises(ValueError):
        model.model_validate({"section_1_1": [paragraph]})
    with pytest.raises(ValueError):
        model.model_validate({"section_1_1": [paragraph], "section_1_2": []})
    assert set(model.model_json_schema()["required"]) == {"section_1_1", "section_1_2"}
