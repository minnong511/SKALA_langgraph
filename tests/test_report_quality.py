"""형식 검사와 LLM Judge의 판정을 결합하는 규칙 검증."""

from unittest.mock import Mock

from kv_cache_agent.agents import report_quality


def _state():
    return {
        "final_report": (
            "# SUMMARY\n비교 결과 [1] [2]\n"
            "## 4.1 기술 성숙도\n내용 [1]\n"
            "## 4.2 시장성\n내용 [2]\n"
            "## 4.3 이해관계자\n내용 [1]\n"
            "## 4.4 도메인 적용\n내용 [2]\n"
            "# REFERENCE\n[1] https://example.org/a\n[2] https://example.org/b"
        ),
        "verified_evidence_cards": [
            {"evidence_id": "a", "source_url": "https://example.org/a"},
            {"evidence_id": "b", "source_url": "https://example.org/b"},
        ],
        "control": {"evidence_ready": True, "report_revision": 1, "max_report_revisions": 2},
    }


def _judge(monkeypatch):
    llm = Mock()
    llm.with_structured_output.return_value.invoke.return_value = {
        "groundedness": True,
        "neutrality": True,
        "bias_control": True,
        "perspective_coverage": True,
        "feedback": [],
    }
    monkeypatch.setattr(report_quality, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(report_quality, "get_llm", lambda: llm)


def test_four_quality_items_pass_when_structure_and_judge_agree(monkeypatch):
    _judge(monkeypatch)
    result = report_quality.evaluate_report_quality(_state())
    assert result["quality_result"]["status"] == "passed"
    assert result["control"]["status"] == "completed"


def test_good_judge_cannot_override_missing_reference(monkeypatch):
    _judge(monkeypatch)
    state = _state()
    state["final_report"] = state["final_report"].replace("[2] https://example.org/b", "")
    result = report_quality.evaluate_report_quality(state)
    assert result["quality_result"]["groundedness"] is False
    assert result["quality_result"]["status"] == "revise"


def test_report_cannot_pass_before_supervisor_approves_evidence(monkeypatch):
    _judge(monkeypatch)
    state = _state()
    state["control"]["evidence_ready"] = False
    result = report_quality.evaluate_report_quality(state)
    assert result["quality_result"]["passed"] is False


def test_failed_generation_skips_judge_and_requests_rewrite(monkeypatch):
    state = _state()
    state["final_report"] = "# 보고서 생성 실패\n작성 오류"
    llm = Mock()
    monkeypatch.setattr(report_quality, "OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(report_quality, "get_llm", llm)
    result = report_quality.evaluate_report_quality(state)
    assert result["quality_result"]["status"] == "revise"
    llm.assert_not_called()
