"""보고서 생성의 목차, 인용, 실패 처리와 노드 연결 테스트.

인풋:
    기존 기술 fixture의 사본과 모의 검증 판정, synthesis_result,
    고정 절별 응답 및 API 오류 Mock.
함수 기능:
    state / response: 보고서 입력과 21개 본문 구역의 응답 준비.
    mock_llm: 실제 보고서 LLM 호출을 Mock으로 교체.
    test_*: 목차, 사용 근거, 입력 보존, 실패 문자열과 종합 노드 연결 확인.
아웃풋:
    pytest의 성공 또는 실패 결과. 실제 API 호출과 PDF 렌더링 없음.
주의:
    정상 Mock 테스트 통과와 실제 생성 문장의 의미적 품질 검증은 별개.

데이터와 함수 형식:
    state() -> dict: user_query, 검증 카드, verification_result,
                     synthesis_result가 포함된 테스트 State.
    response(state: dict) -> dict: {"sections": list[dict]} 형태의 LLM 모의 응답.
    각 section: {"section_id": str, "paragraphs": list[dict]}.
    각 paragraph: {"text": str, "claim_type": str, "evidence_ids": list[str]}.
    mock_llm(monkeypatch, reply: dict) -> tuple[Mock, Mock]:
        로더 Mock과 LLM Mock 반환.
    test_*(...) -> None: assert 성공 시 정상 종료, 실패 시 예외로 pytest에 전달.
    검사 대상 반환 형식: {"final_report": str}.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from kv_cache_agent.agents import report_writer as module


@pytest.fixture
def state():
    """보고서 테스트용 검증 카드와 기술 종합 결과 준비. 실제 검증 수행은 아님."""
    state = json.loads(
        (Path(__file__).parent / "fixtures/sample_technical_result.json").read_text()
    )
    for card in state["evidence_cards"]:
        card["verification_status"] = "verified"
    state["verification_result"] = {"status": "ok"}
    state["user_query"] = "클라우드 비용과 지연 비교"
    state["synthesis_result"] = {
        "status": "insufficient_evidence",
        "summary": "기술 근거만 확보",
        "evidence_ids": [c["evidence_id"] for c in state["evidence_cards"]],
        "limitations": ["시장 정보 부족"],
        "payload": {},
    }
    return state


def response(state):
    """SUMMARY와 하위 절 20개에 동일한 가상 문단을 넣는 형식 검사 전용 응답."""
    return {
        "sections": [
            {
                "section_id": s["id"],
                "paragraphs": [
                    {
                        "text": "조건에 한정한 기술 해석",
                        "claim_type": "inference",
                        "evidence_ids": [state["evidence_cards"][2]["evidence_id"]],
                    }
                ],
            }
            for s in module._body_sections(module._load_prompt_config())
        ]
    }


def mock_llm(monkeypatch, reply):
    """보고서 모듈의 get_llm을 교체하여 실제 호출 없이 지정 응답 반환."""
    llm = Mock()
    # 실제 호출 체인의 invoke 결과를 준비한 고정 응답으로 대체.
    llm.with_structured_output.return_value.invoke.return_value = reply
    loader = Mock(return_value=llm)
    # 호출하는 모듈의 이름을 교체하며, 테스트 종료 후 pytest가 자동 복원.
    monkeypatch.setattr(module, "get_llm", loader)
    return loader, llm


def test_outline_citations_and_input_preservation(state, monkeypatch):
    """목차 순서, 실제 사용 출처만 기재, 미상 날짜, 한계와 입력 보존 확인."""
    # 중첩된 카드까지 복사하여 실행 후 입력 변경 여부 비교.
    before = deepcopy(state)
    _, llm = mock_llm(monkeypatch, response(state))
    output = module.report_writer_agent(state)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert set(output) == {"final_report"}
    text = output["final_report"]
    assert re.findall(r"^# (.+)$", text, re.MULTILINE) == module.TITLES
    assert len(re.findall(r"^## ", text, re.MULTILINE)) == 20
    refs = text.split("# REFERENCE\n")[1]
    assert "technical-cxl_based-001" in refs
    assert "technical-turboquant-001" not in refs
    assert "발행일 미상" in refs and "p. 1" in refs
    assert "시장 정보 부족" in text
    assert state == before
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert state["user_query"] in messages[1].content


def test_empty_evidence_no_api(monkeypatch):
    """근거가 없으면 LLM 없이 고정 목차의 판단 보류 안내를 만드는지 확인."""
    loader, _ = mock_llm(monkeypatch, {})
    output = module.report_writer_agent(
        {"synthesis_result": {"status": "insufficient_evidence", "evidence_ids": []}}
    )
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "판단 보류" in output["final_report"]
    assert (
        re.findall(r"^# (.+)$", output["final_report"], re.MULTILINE) == module.TITLES
    )
    loader.assert_not_called()


@pytest.mark.parametrize(
    "mode", ["unknown", "uncited", "missing", "duplicate", "heading"]
)
def test_bad_generation_rejected(state, monkeypatch, mode):
    """잘못된 ID, 무인용 주장, 누락 절, 중복 절, 임의 제목의 다섯 오류 차단 확인."""
    reply = response(state)
    paragraph = reply["sections"][0]["paragraphs"][0]
    if mode == "unknown":
        paragraph["evidence_ids"] = ["invented"]
    elif mode == "uncited":
        paragraph["evidence_ids"] = []
    elif mode == "missing":
        reply["sections"].pop()
    elif mode == "duplicate":
        reply["sections"].append(reply["sections"][0])
    else:
        paragraph["text"] = "# 가짜 제목"
    mock_llm(monkeypatch, reply)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.report_writer_agent(state)["final_report"].startswith(
        "# 보고서 생성 실패"
    )


def test_unverified_reference_rejected_before_api(state, monkeypatch):
    """종합 결과가 미검증 카드를 인용하면 보고서 LLM 호출 전에 차단하는지 확인."""
    state["evidence_cards"][0]["verification_status"] = "unverified"
    loader, _ = mock_llm(monkeypatch, {})
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]
    loader.assert_not_called()


def test_api_failure_keeps_string_and_redacts(state, monkeypatch):
    """API 실패에도 문자열 반환 계약 유지 및 원본 오류 메시지 노출 방지 확인."""
    _, llm = mock_llm(monkeypatch, {})
    # 정상 응답 대신 예외 발생을 모의. SECRET은 노출 검사 전용 가짜 문자열.
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("SECRET")
    result = module.report_writer_agent(state)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert isinstance(result["final_report"], str)
    assert "생성 실패" in result["final_report"]
    assert "SECRET" not in result["final_report"]


def test_invalid_outline(state, monkeypatch, tmp_path):
    """빈 목차 YAML의 설정 오류를 실패 안내 문자열로 처리하는지 확인."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "system_prompt: test\nconstraints:\n  allow_new_search: false\nreport_structure: []"
    )
    monkeypatch.setattr(module, "PROMPT_PATH", path)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]


def test_fabricated_number_rejected(state, monkeypatch):
    """근거에 없는 98765라는 숫자를 생성한 응답의 차단 확인."""
    reply = response(state)
    reply["sections"][0]["paragraphs"][0]["text"] = "처리량 98765% 증가"
    mock_llm(monkeypatch, reply)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]


def test_synthesis_to_report_with_mocked_llms(state, monkeypatch):
    """두 에이전트를 순서대로 호출하여 종합 결과, 한계, 근거의 보고서 전달 확인."""
    from kv_cache_agent.agents import synthesis

    key = state["evidence_cards"][0]["evidence_id"]
    item = {"text": "테스트용 해석", "evidence_ids": [key], "claim_type": "inference"}
    llm = Mock()
    # 실제 호출 체인의 invoke 결과를 준비한 고정 응답으로 대체.
    llm.with_structured_output.return_value.invoke.return_value = {
        "summary": [item],
        "comparison_rows": [{**item, "perspective": "technical"}],
        "agreements": [],
        "conflicts": [],
        "conditional_recommendations": [],
        "limitations": ["다른 관점의 자료 부족"],
    }
    monkeypatch.setattr(synthesis, "get_llm", Mock(return_value=llm))
    # LangGraph의 갱신처럼 종합 반환값을 새 State에 합친 뒤 보고서 노드에 전달.
    updated = {**state, **synthesis.synthesis_agent(state)}
    reply = response(state)
    for section in reply["sections"]:
        section["paragraphs"][0]["evidence_ids"] = [key]
    mock_llm(monkeypatch, reply)
    # 실행: 외부 API 없이 종합 결과를 보고서 문자열로 변환.
    report = module.report_writer_agent(updated)["final_report"]
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert report.startswith("# SUMMARY")
    assert "다른 관점의 자료 부족" in report
    assert key in report


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "success",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "render",
                "validate_report",
            ],
        ),
        ("empty", ["load_config", "prepare_context", "fallback", "validate_report"]),
        ("upstream_failed", ["load_config", "prepare_context", "failure"]),
        (
            "bad_citation",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "failure",
            ],
        ),
        ("api_error", ["load_config", "prepare_context", "generate", "failure"]),
        (
            "render_error",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "render",
                "failure",
            ],
        ),
    ],
)
def test_report_graph_routes(state, monkeypatch, mode, expected):
    """생성과 렌더링의 실제 노드 실행 및 오류 시 후속 노드 미실행 확인."""
    reply = response(state)
    if mode == "bad_citation":
        reply["sections"][0]["paragraphs"][0]["evidence_ids"] = ["missing"]
    loader, llm = mock_llm(monkeypatch, reply)
    if mode == "empty":
        state["synthesis_result"]["evidence_ids"] = []
    if mode == "upstream_failed":
        state["synthesis_result"]["status"] = "failed"
    if mode == "api_error":
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
            "SECRET"
        )
    if mode == "render_error":
        monkeypatch.setattr(
            module, "_render_markdown", Mock(side_effect=RuntimeError("SECRET"))
        )
    before = deepcopy(state)
    events = list(
        module.build_report_graph().stream({"request": state}, stream_mode="updates")
    )
    assert [name for event in events for name in event] == expected
    output = events[-1][expected[-1]]["output"]
    assert set(output) == {"final_report"}
    assert isinstance(output["final_report"], str)
    assert "SECRET" not in str(events)
    assert state == before
    if mode in ("empty", "upstream_failed"):
        loader.assert_not_called()
    else:
        llm.with_structured_output.return_value.invoke.assert_called_once()


def test_report_cached_graph_recovers_after_error(state, monkeypatch):
    """이전 실행의 실패 상태가 재사용 그래프의 다음 정상 실행에 영향을 주지 않는지 확인."""
    _, llm = mock_llm(monkeypatch, response(state))
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("SECRET")
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]
    llm.with_structured_output.return_value.invoke.side_effect = None
    assert module.report_writer_agent(state)["final_report"].startswith("# SUMMARY")
