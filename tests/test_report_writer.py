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
    state["verified_evidence_cards"] = deepcopy(state["evidence_cards"])
    state["usable_evidence_cards"] = deepcopy(state["evidence_cards"])
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
    assert text.index("시장 정보 부족") > text.index("## 6.4")
    assert text.index(module.ANALYTICAL_METHOD_NOTE) > text.index("## 6.4")
    assert state == before
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert state["user_query"] in messages[1].content


def test_report_writer_uses_all_usable_cards(state, monkeypatch):
    """종합 핵심 카드에 없는 허용 카드도 보고서 작성에 사용할 수 있는지 확인."""
    state["synthesis_result"]["evidence_ids"] = [
        state["evidence_cards"][0]["evidence_id"]
    ]
    reply = response(state)
    mock_llm(monkeypatch, reply)

    report = module.report_writer_agent(state)["final_report"]

    assert report.startswith("# SUMMARY")
    messages = module.get_llm.return_value.with_structured_output.return_value.invoke
    context = messages.call_args.args[0][1].content
    assert state["evidence_cards"][2]["evidence_id"] in context


def test_redundant_inline_evidence_metadata_is_removed(state, monkeypatch):
    """본문에 중복된 evidence_ids 메타데이터가 있어도 별도 필드 근거를 사용해 생성하는지 확인."""
    reply = response(state)
    paragraph = reply["sections"][0]["paragraphs"][0]
    paragraph["text"] = (
        "추론: "
        + paragraph["text"]
        + " (evidence_ids: [technical-cxl_based-001])"
    )
    mock_llm(monkeypatch, reply)

    report = module.report_writer_agent(state)["final_report"]

    assert report.startswith("# SUMMARY")
    assert "evidence_ids:" not in report
    assert "추론:" not in report


def test_empty_evidence_generates_detailed_analysis(monkeypatch):
    """근거가 없어도 LLM을 실행하고 모든 절을 내용으로 채우는지 확인."""
    reply = {
        "sections": [
            {
                "section_id": section["id"],
                "paragraphs": [
                    {
                        "text": "직접 근거는 부족하지만 기술 특성상 가능성이 있다.",
                        "claim_type": "inference",
                        "evidence_ids": [],
                    }
                ],
            }
            for section in module._body_sections(module._load_prompt_config())
        ]
    }
    loader, _ = mock_llm(monkeypatch, reply)
    output = module.report_writer_agent(
        {"synthesis_result": {"status": "insufficient_evidence", "evidence_ids": []}}
    )
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "추론:" not in output["final_report"]
    assert output["final_report"].count(module.ANALYTICAL_METHOD_NOTE) == 1
    assert "검증 가능한 근거가 부족하여 이 항목의 판단을 보류한다." not in output[
        "final_report"
    ]
    assert (
        re.findall(r"^# (.+)$", output["final_report"], re.MULTILINE) == module.TITLES
    )
    loader.assert_called_once()
    context = loader.return_value.with_structured_output.return_value.invoke.call_args.args[
        0
    ][1].content
    assert '"inference_detail_mode": true' in context
    assert "기술 원리에서 출발해 작동 메커니즘을 설명한다." in context


@pytest.mark.parametrize("mode", ["heading"])
def test_bad_generation_rejected(state, monkeypatch, mode):
    """보고서 구조 자체가 잘못된 경우에만 생성을 중단하는지 확인."""
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


@pytest.mark.parametrize("mode", ["missing", "duplicate"])
def test_section_shape_is_normalized(state, monkeypatch, mode):
    """누락 절은 추론으로, 중복 절은 병합으로 보완한 뒤 보고서를 생성하는지 확인."""
    reply = response(state)
    if mode == "missing":
        reply["sections"].pop()
    else:
        reply["sections"].append(deepcopy(reply["sections"][0]))
    mock_llm(monkeypatch, reply)

    report = module.report_writer_agent(state)["final_report"]

    assert report.startswith("# SUMMARY")
    if mode == "missing":
        assert "남은 미확인 사항은 실제 클라우드 워크로드에서의 품질 변화" in report
        assert report.count(module.ANALYTICAL_METHOD_NOTE) == 1
    else:
        assert "조건에 한정한 기술 해석" in report


def test_uncited_fact_is_downgraded_to_inference(state, monkeypatch):
    """자료 부족 상태의 무인용 fact를 사실로 통과시키지 않고 추론으로 낮추는지 확인."""
    reply = response(state)
    paragraph = reply["sections"][0]["paragraphs"][0]
    paragraph["claim_type"] = "fact"
    paragraph["evidence_ids"] = []
    mock_llm(monkeypatch, reply)

    report = module.report_writer_agent(state)["final_report"]

    assert report.startswith("# SUMMARY")
    assert "추론:" not in report
    assert module.ANALYTICAL_METHOD_NOTE in report


def test_unverified_reference_rejected_before_api(state, monkeypatch):
    """종합 결과가 미검증 카드를 인용하면 보고서 LLM 호출 전에 차단하는지 확인."""
    state["evidence_cards"][0]["verification_status"] = "unverified"
    state["verified_evidence_cards"][0]["verification_status"] = "unverified"
    state["usable_evidence_cards"][0]["verification_status"] = "unverified"
    loader, _ = mock_llm(monkeypatch, {})
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]
    loader.assert_not_called()


def test_partial_cards_can_generate_provisional_report(state, monkeypatch):
    """부분 검증 카드만 있어도 잠정 보고서를 생성하는지 확인."""
    state["verification_result"] = {
        "status": "insufficient_evidence",
        "limitations": ["부분 검증 근거를 포함한 잠정 평가"],
    }
    state["synthesis_result"]["status"] = "insufficient_evidence"
    for card in state["evidence_cards"]:
        card["verification_status"] = "partially_verified"
    state["verified_evidence_cards"] = []
    state["usable_evidence_cards"] = deepcopy(state["evidence_cards"])

    loader, _ = mock_llm(monkeypatch, response(state))
    report = module.report_writer_agent(state)["final_report"]

    assert report.startswith("# SUMMARY")
    assert "부분 검증 근거" in report
    loader.assert_called_once()


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


def test_unknown_citation_is_downgraded_to_inference(state, monkeypatch):
    """잘못된 인용 ID가 있어도 해당 문장을 추론으로 바꿔 보고서를 생성하는지 확인."""
    reply = response(state)
    reply["sections"][0]["paragraphs"][0]["evidence_ids"] = ["invented"]
    mock_llm(monkeypatch, reply)

    result = module.report_writer_agent(state)

    assert result["final_report"].startswith("# SUMMARY")
    assert "추론:" not in result["final_report"]


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
    """근거에 없는 숫자도 추론 문장으로 보고서에 포함하는지 확인."""
    reply = response(state)
    reply["sections"][0]["paragraphs"][0]["text"] = "처리량 98765% 증가"
    mock_llm(monkeypatch, reply)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    report = module.report_writer_agent(state)["final_report"]
    assert report.startswith("# SUMMARY")
    assert "처리량 98765% 증가" in report
    assert "추론:" not in report


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
        (
            "empty",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "render",
                "validate_report",
            ],
        ),
        (
            "upstream_failed",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "render",
                "validate_report",
            ],
        ),
        (
            "bad_citation",
            [
                "load_config",
                "prepare_context",
                "generate",
                "validate_sections",
                "render",
                "validate_report",
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
    _loader, llm = mock_llm(monkeypatch, reply)
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
    llm.with_structured_output.return_value.invoke.assert_called_once()


def test_report_cached_graph_recovers_after_error(state, monkeypatch):
    """이전 실행의 실패 상태가 재사용 그래프의 다음 정상 실행에 영향을 주지 않는지 확인."""
    _, llm = mock_llm(monkeypatch, response(state))
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("SECRET")
    assert "생성 실패" in module.report_writer_agent(state)["final_report"]
    llm.with_structured_output.return_value.invoke.side_effect = None
    assert module.report_writer_agent(state)["final_report"].startswith("# SUMMARY")
