"""Keep deferred defects visible until their planned migration stage."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest


def test_legacy_pdf_cache_uses_each_cards_locator():
    from kv_cache_agent.agents import verifier

    paper = Path(__file__).parents[1] / "data/papers/turboquant.pdf"
    cards = [
        {
            "evidence_id": f"p{p}",
            "source_url": str(paper),
            "source_type": "paper",
            "source_locator": f"p. {p}",
        }
        for p in (2, 17)
    ]
    result = verifier._recheck_original_sources_node(
        {"cards": cards, "active_evidence_ids": ["p2", "p17"]}
    )
    assert (
        result["source_documents"]["p2"]["content"]
        != result["source_documents"]["p17"]["content"]
    )


@pytest.mark.xfail(
    strict=True, reason="Stage 8: cloud worker locator ownership migration"
)
def test_cloud_worker_preserves_requested_page():
    from kv_cache_agent.agents import cloud_domain as module

    path = "data/papers/turboquant.pdf"
    extraction = module.CloudDomainExtraction(
        summary="test",
        comparison=module.CloudComparison(turboquant="x", cxl_based="y", trade_off="z"),
        findings=[
            module.CloudDomainFinding(
                technology="TurboQuant",
                criterion="gpu_memory_cost",
                claim="Memory reduced",
                evidence_text="quote",
                source_url=path,
                source_locator="p. 2",
                source_type="paper",
                claim_type="fact",
                confidence=0.9,
            )
        ],
    )
    technical = [
        {"source_url": path, "source_type": "paper", "source_locator": f"p. {p}"}
        for p in (2, 17)
    ]
    cards, _ = module._build_evidence_cards(extraction, technical, [], [])
    assert cards[0]["source_locator"] == "p. 2"


@pytest.mark.xfail(
    strict=True, reason="Stage 11: legacy report writer validation migration"
)
def test_legacy_report_writer_must_reject_fabricated_numbers(monkeypatch):
    from kv_cache_agent.agents import report_writer as module

    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "sections": [
            {
                "section_id": s["id"],
                "paragraphs": [
                    {
                        "text": "비용 987654321% 감소",
                        "claim_type": "fact",
                        "evidence_ids": [],
                    }
                ],
            }
            for s in module._body_sections(module._load_prompt_config())
        ]
    }
    monkeypatch.setattr(module, "get_llm", lambda: llm)
    report = module.report_writer_agent(
        {
            "synthesis_result": {"status": "insufficient_evidence", "evidence_ids": []},
            "verification_result": {"status": "insufficient_evidence"},
            "usable_evidence_cards": [],
        }
    )["final_report"]
    assert "987654321%" not in report


@pytest.mark.xfail(
    strict=True, reason="Stage 12: CLI result status/exit code migration"
)
def test_legacy_cli_does_not_label_failed_report_completed(monkeypatch, tmp_path):
    import sys

    from kv_cache_agent import main as module

    workflow = Mock()
    workflow.invoke.return_value = {
        "final_report": "# 보고서 생성 실패",
        "synthesis_result": {"status": "failed"},
    }
    monkeypatch.setattr(module, "build_workflow", lambda: workflow)
    monkeypatch.setattr(module, "OUTPUTS_DIR", tmp_path)
    monkeypatch.setattr(module, "write_pdf", Mock())
    log = tmp_path / "log.json"
    monkeypatch.setattr(sys, "argv", ["test", "--log-file", str(log)])
    module.main()
    assert json.loads(log.read_text())["status"] == "failed"
