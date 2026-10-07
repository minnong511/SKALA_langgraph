"""Search adaptation, source independence and measurement safeguards."""

from copy import deepcopy
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from kv_cache_agent.agents import verifier
from kv_cache_agent.agents.compat import compact_verification
from kv_cache_agent.agents.quality_evaluator import check_report_rules
from kv_cache_agent.agents.research_worker import _research_once, research_worker
from kv_cache_agent.evidence import (
    claim_fingerprint,
    comparison_context,
    source_identity,
)
from kv_cache_agent.mock_run import mock_plan, mock_services
from kv_cache_agent.schemas.outputs import WorkerResult
from kv_cache_agent.schemas.technical import TechnicalFinding
from kv_cache_agent.tools.tavily_search import search_web


def card(**updates):
    return {
        "evidence_id": "e1",
        "technology": "CXL-based",
        "perspective": "technical",
        "claim": "ITME improves throughput by 1.80x over NVMe-oF.",
        "evidence_text": "ITME improves throughput by 1.80x over NVMe-oF.",
        "source_title": "ITME",
        "source_url": "data/papers/cxl_based_kv_cache.pdf",
        "source_type": "paper",
        "source_locator": "p. 2",
        "retrieval_method": "faiss",
        "published_date": "2026-06",
        "claim_type": "fact",
        "confidence": 0.9,
        "caveat": "Reported experiment only",
        "verification_status": "verified",
        **updates,
    }


def request():
    task = (
        mock_plan("simple")
        .tasks[0]
        .model_copy(
            update={
                "search_queries": [
                    "primary query",
                    "alternate official query",
                    "independent limitations query",
                ],
                "query": "primary query",
                "preferred_domains": ["arxiv.org"],
            }
        )
    )
    return {
        "task": task.model_dump(),
        "trace_id": "test-trace",
        "user_query": "평가",
        "target_domain": "LLM 서빙",
        "limits": {"max_worker_retries": 2, "max_cards_per_task": 6},
    }


def result(task, cards, **updates):
    return WorkerResult(
        task_id=task.task_id,
        technology=task.technology,
        perspective=task.perspective,
        status="success" if cards else "failed",
        evidence_cards=cards,
        failure_kind="none" if cards else "insufficient_evidence",
        **updates,
    )


def test_insufficient_evidence_changes_query_and_relaxes_domain():
    observed = []

    def service(task, req):
        observed.append((task.query, req["restrict_domains"]))
        return result(
            task, [] if task.retry_count == 0 else [card(technology="TurboQuant")]
        )

    row = research_worker(request(), service)["payload"]["worker_results"][0]
    assert observed == [("primary query", True), ("alternate official query", False)]
    assert row["status"] == "success" and row["retry_count"] == 1
    assert [r["query"] for r in row["search_attempts"]] == [q for q, _ in observed]


def test_transport_failure_retries_same_query():
    observed = []

    def service(task, req):
        observed.append(task.query)
        if task.retry_count == 0:
            raise TimeoutError("sensitive body")
        return result(task, [card(technology="TurboQuant")])

    row = research_worker(request(), service)["payload"]["worker_results"][0]
    assert observed == ["primary query", "primary query"]
    assert row["search_attempts"][0]["failure_kind"] == "transport"
    assert "sensitive" not in str(row)


def test_duplicate_evidence_triggers_different_search_instead_of_false_success():
    req = request()
    previous = card(technology="TurboQuant")
    req["known_evidence_keys"] = [claim_fingerprint(previous)]
    observed = []

    def service(task, _req):
        observed.append(task.query)
        return result(
            task,
            [previous]
            if not task.retry_count
            else [
                card(technology="TurboQuant", claim="A separate documented limitation")
            ],
        )

    row = research_worker(req, service)["payload"]["worker_results"][0]
    assert observed == ["primary query", "alternate official query"]
    assert row["search_attempts"][0]["duplicate_cards"] == 1
    assert len(row["evidence_cards"]) == 1


def test_all_empty_attempts_are_bounded_and_return_no_findings():
    row = research_worker(request(), lambda task, _: result(task, []))["payload"][
        "worker_results"
    ][0]
    assert len(row["search_attempts"]) == 3
    assert len({r["query"] for r in row["search_attempts"]}) == 3
    assert row["status"] == "failed" and row["evidence_cards"] == row["findings"] == []


def test_tavily_receives_domain_filter_and_advanced_depth():
    client = Mock()
    client.search.return_value = {"results": []}
    search_web(
        "support",
        include_domains=["research.google"],
        search_depth="advanced",
        client=client,
    )
    assert client.search.call_args.kwargs["include_domains"] == ["research.google"]
    assert client.search.call_args.kwargs["search_depth"] == "advanced"


def test_worker_uses_raw_source_and_scoped_advanced_search(monkeypatch):
    from kv_cache_agent.agents import research_worker as module

    with mock_services():
        search = Mock(wraps=module.search_web)
        monkeypatch.setattr(module, "search_web", search)
        req = request()
        task = (
            mock_plan("simple")
            .tasks[2]
            .model_copy(update={"preferred_domains": ["research.google"]})
        )
        output = _research_once(task, {**req, "restrict_domains": True})
    assert search.call_args.kwargs["include_domains"] == ["research.google"]
    assert search.call_args.kwargs["search_depth"] == "advanced"
    assert output.search_attempts[0]["source_count"] == 2
    assert all(c["source_chunk_ids"] for c in output.evidence_cards)


@pytest.mark.parametrize(
    "url",
    [
        "data/papers/cxl_based_kv_cache.pdf",
        "https://arxiv.org/abs/2606.12556",
        "https://arxiv.org/html/2606.12556v2",
        "https://arxiv.org/pdf/2606.12556v1",
    ],
)
def test_same_paper_has_one_source_identity(url):
    assert source_identity({"source_url": url}) == "arxiv:2606.12556"


def test_url_tracking_does_not_add_an_independent_source():
    assert (
        source_identity(
            {"source_url": "http://www.example.com/page/?utm_source=a#section"}
        )
        == "https://example.com/page"
    )


def measurement(**updates):
    return {
        **card(
            metric="throughput improvement",
            value="1.80",
            unit="x",
            baseline="NVMe-oF",
            conditions={
                "model": "Llama3",
                "hardware": "A100",
                "workload": "extended-turns",
            },
        ),
        **updates,
    }


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"baseline": "CPU-offload"}, "different_baseline"),
        ({"metric": "batch size", "unit": "%"}, "different_metric_or_unit"),
        ({"conditions": {}}, "unknown_experimental_conditions"),
    ],
)
def test_different_baselines_or_metrics_are_not_directly_comparable(change, reason):
    left = measurement()
    right = {**deepcopy(left), "evidence_id": "e2", "value": "35.7", **change}
    pair = comparison_context([left, right])[0]
    assert pair["directly_comparable"] is False and reason in pair["reasons"]
    assert "conflict" not in pair


def test_same_explicit_conditions_allow_comparison():
    left = measurement()
    assert comparison_context([left, {**left, "evidence_id": "e2", "value": "1.70"}])[
        0
    ]["directly_comparable"]


def test_measurement_survives_structured_extraction_and_result_schema():
    finding = TechnicalFinding(
        technology="CXL-based",
        claim="Measured result",
        evidence_text="1.80x over NVMe-oF",
        source_chunk_ids=["chunk"],
        claim_type="fact",
        confidence=0.9,
        metric="throughput",
        value="1.80",
        unit="x",
        baseline="NVMe-oF",
        conditions={"model": "Llama3"},
    )
    assert finding.conditions.hardware is None
    row = result(mock_plan("simple").tasks[0], [measurement()]).model_dump()
    assert row["evidence_cards"][0]["baseline"] == "NVMe-oF"


def test_report_source_counts_do_not_double_count_pdf_and_arxiv():
    payload = {
        "usable_evidence_cards": [
            card(),
            card(evidence_id="e2", source_url="https://arxiv.org/html/2606.12556v2"),
        ]
    }
    rules = check_report_rules(payload)
    assert rules["source_counts"] == {"arxiv:2606.12556": 2} and rules["single_source"]


def test_compact_verification_keeps_audit_but_deduplicates_support():
    original = measurement()
    duplicate = {
        **original,
        "evidence_id": "e2",
        "source_url": "https://arxiv.org/html/2606.12556v2",
        "verification_status": "partially_verified",
    }
    result = {
        "verification_result": {},
        "verified_evidence_cards": [original, duplicate],
        "usable_evidence_cards": [duplicate, original],
    }
    compact = compact_verification(result)
    assert compact["usable_evidence_cards"] == [original]
    assert len(compact["verified_evidence_cards"]) == 2


@pytest.mark.parametrize("value,support", [("1.80", "full"), ("35.7", "none")])
def test_structured_numbers_must_exist_in_literal_quote(monkeypatch, value, support):
    candidate = measurement(value=value)
    llm = Mock()
    llm.with_structured_output.return_value.batch.return_value = [
        {
            "decisions": [
                {
                    "evidence_id": "e1",
                    "support_level": "full",
                    "matched_quotes": [candidate["evidence_text"]],
                    "rationale": "supported",
                    "claim_type_assessment": "correct",
                }
            ]
        }
    ]
    monkeypatch.setattr(verifier, "get_llm", lambda: llm)
    decision = verifier._compare_claims_with_sources(
        [candidate], {"e1": {"content": candidate["evidence_text"]}}
    ).decisions[0]
    assert decision.support_level == support


def test_planner_rejects_url_as_domain_and_overlong_search_query():
    task = mock_plan("simple").tasks[0]
    for updates in (
        {"preferred_domains": ["https://arxiv.org"]},
        {"search_queries": ["x" * 181]},
    ):
        with pytest.raises(ValidationError):
            type(task).model_validate({**task.model_dump(), **updates})


def test_numeric_check_reads_verified_measurement_conditions():
    from kv_cache_agent.agents.synthesis import _validate_numbers

    candidate = measurement()
    candidate["conditions"] = {
        "model": "Llama-3.1-8B-Instruct",
        "workload": "4k~104k tokens",
    }
    _validate_numbers(
        "Llama-3.1-8B-Instruct의 4k~104k 조건에서 1.8배를 보고했다.",
        ["e1"],
        {"e1": candidate},
    )
    with pytest.raises(ValueError, match="Numeric claim absent"):
        _validate_numbers("같은 조건에서 99배를 보고했다.", ["e1"], {"e1": candidate})


def test_writer_removes_only_inline_ids_attached_to_same_paragraph():
    from kv_cache_agent.agents.report_writer import _remove_inline_evidence_artifact

    assert (
        _remove_inline_evidence_artifact(
            "확인한 내용 [task:e1, task:e2]", ["task:e1", "task:e2"]
        )
        == "확인한 내용"
    )
    assert "[unknown]" in _remove_inline_evidence_artifact(
        "확인한 내용 [unknown]", ["task:e1"]
    )
    assert "[task:e2]" in _remove_inline_evidence_artifact(
        "확인한 내용 [task:e2]", ["task:e1"]
    )
