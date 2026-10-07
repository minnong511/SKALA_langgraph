"""Supervisor 내용 검수는 하위 에이전트 상태와 별개로 근거를 검토해야 한다."""

from unittest.mock import Mock

from kv_cache_agent.agents import supervisor


def _state():
    return {
        "verified_evidence_cards": [
            {
                "evidence_id": p,
                "perspective": p,
                "technology": "CXL-based" if p == "cloud_domain" else "TurboQuant",
                "source_url": f"https://example.org/{p}",
                "claim": "제공된 주장",
                "evidence_text": "제공된 원문",
            }
            for p in supervisor.WORKERS
        ],
        "market_result": {"status": "insufficient_evidence"},
    }


def _reply():
    return {
        "reviews": [
            {
                "perspective": p,
                "sufficient": True,
                "reason": "평가 범위를 지지하는 실제 근거 확보",
                "supporting_evidence_ids": [p],
                "missing_information": [],
            }
            for p in supervisor.WORKERS
        ]
    }


def _mock(monkeypatch, reply):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = reply
    monkeypatch.setattr(supervisor, "OPENAI_API_KEY", "test")
    monkeypatch.setattr(supervisor, "get_llm", lambda: llm)
    return llm


def test_supervisor_owns_scope_approval_even_if_worker_has_minor_gaps(monkeypatch):
    llm = _mock(monkeypatch, _reply())
    gaps, review = supervisor._review_evidence(
        _state(), {"market": ["전체 세부 항목 미확보"]}
    )
    assert not gaps and len(review) == 4
    context = llm.with_structured_output.return_value.invoke.call_args.args[0][
        1
    ].content
    assert "evidence_text" in context


def test_actual_content_gap_overrides_worker_completion(monkeypatch):
    reply = _reply()
    reply["reviews"][1].update(
        sufficient=False, missing_information=["직접 채택과 일반 시장 지표 구분 필요"]
    )
    _mock(monkeypatch, reply)
    gaps, _ = supervisor._review_evidence(_state(), {})
    assert "market" in gaps


def test_unknown_review_citations_never_approve(monkeypatch):
    reply = _reply()
    reply["reviews"][0]["supporting_evidence_ids"] = ["made-up"]
    _mock(monkeypatch, reply)
    gaps, reviews = supervisor._review_evidence(_state(), {})
    assert gaps and not reviews
