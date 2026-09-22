"""평가 종합의 검증 경계와 입출력 계약 테스트.

인풋:
    fixtures/sample_technical_result.json의 미검증 기술 카드 4개.
    테스트별 사본, 가상 관점별 카드, 고정 LLM 응답과 오류 Mock.
함수 기능:
    state / verified / draft: 기본 입력과 검증 판정 사본, 생성 응답 준비.
    install_llm: 실제 API 호출을 Mock으로 교체.
    test_*: 근거 선별, 관점별 종합, 오류와 인용 검사, 입력 보존 확인.
아웃풋:
    pytest의 성공 또는 실패 결과. 실제 보고서 파일과 외부 API 호출 없음.
주의:
    사본의 verified 판정은 테스트 설정이며 실제 논문 검증 완료를 의미하지 않음.

데이터와 함수 형식:
    원본 JSON: {"technical_result": AgentResult,
                "evidence_cards": list[EvidenceCard]}
    state() -> dict: user_query와 verification_result가 추가된 테스트 입력.
    verified(state: dict) -> dict: 원본 변경 없는 모의 검증 상태 사본.
    draft(state: dict) -> dict: SynthesisDraft 모델에 대응하는 고정 응답.
    install_llm(monkeypatch, response: dict) -> tuple[Mock, Mock]:
        로더 Mock과 LLM Mock 반환.
    test_*(...) -> None: assert 성공 시 정상 종료, 실패 시 예외로 pytest에 전달.
    검사 대상 반환 형식: {"synthesis_result": AgentResult}.
"""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import pytest

from kv_cache_agent.agents import synthesis as module

FIXTURE = Path(__file__).parent / "fixtures" / "sample_technical_result.json"


@pytest.fixture
def state():
    """테스트마다 원본 JSON을 새로 읽어 입력 준비. 카드 판정은 unverified 유지."""
    result = json.loads(FIXTURE.read_text())
    result["user_query"] = "클라우드의 비용과 지연시간 비교"
    result["verification_result"] = {"status": "ok", "evidence_ids": []}
    return result


def verified(state):
    """원본을 복사한 뒤 카드 판정만 verified로 변경하는 테스트용 보조 함수."""
    state = deepcopy(state)
    for card in state["evidence_cards"]:
        card["verification_status"] = "verified"
    state["verified_evidence_cards"] = deepcopy(state["evidence_cards"])
    state["usable_evidence_cards"] = deepcopy(state["evidence_cards"])
    return state


def draft(state):
    """실제 API 대신 반환할 종합 응답 구성. 기술 관점만 포함한 불완전 결과."""
    ids = [c["evidence_id"] for c in state["evidence_cards"]]
    item = {
        "text": "테스트용 조건부 해석",
        "evidence_ids": ids,
        "claim_type": "inference",
    }
    return {
        "summary": [item],
        "comparison_rows": [{**item, "perspective": "technical"}],
        "agreements": [],
        "conflicts": [],
        "conditional_recommendations": [item],
        "limitations": ["운영 비용 판단 보류"],
    }


def install_llm(monkeypatch, response):
    """get_llm을 가짜 로더로 교체하고 호출 여부 확인용 Mock 두 개 반환."""
    llm = Mock()
    # 실제 호출 체인의 invoke 결과를 준비한 고정 응답으로 대체.
    llm.with_structured_output.return_value.invoke.return_value = response
    loader = Mock(return_value=llm)
    # 호출하는 모듈의 이름을 교체하며, 테스트 종료 후 pytest가 자동 복원.
    monkeypatch.setattr(module, "get_llm", loader)
    return loader, llm


def test_original_fixture_is_not_verified(state, monkeypatch):
    """검증 단계가 ok여도 미검증 카드는 사용하지 않고 입력을 보존하는지 확인."""
    # 중첩된 카드까지 복사하여 실행 후 입력 변경 여부 비교.
    before = deepcopy(state)
    loader, _ = install_llm(monkeypatch, {})
    output = module.synthesis_agent(state)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert set(output) == {"synthesis_result"}
    assert output["synthesis_result"]["status"] == "insufficient_evidence"
    assert output["synthesis_result"]["evidence_ids"] == []
    loader.assert_not_called()
    assert state == before


def test_technical_only_preserves_limits_and_query(state, monkeypatch):
    """기술 근거만으로 전체 완료 판정 금지, 사용자 질문 전달과 입력 보존 확인."""
    state = verified(state)
    # 중첩된 카드까지 복사하여 실행 후 입력 변경 여부 비교.
    before = deepcopy(state)
    _, llm = install_llm(monkeypatch, draft(state))
    output = module.synthesis_agent(state)["synthesis_result"]
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert output["status"] == "insufficient_evidence"
    assert len(output["evidence_ids"]) == 4
    assert output["payload"]["comparison_rows"]
    messages = llm.with_structured_output.return_value.invoke.call_args.args[0]
    assert state["user_query"] in messages[1].content
    assert state == before


def test_synthesis_consumes_explicit_global_verified_cards(state, monkeypatch):
    """원본 카드가 미검증이어도 GlobalState의 검증 카드만 종합에 사용하는지 확인."""
    verified_cards = deepcopy(state["evidence_cards"])
    for card in verified_cards:
        card["verification_status"] = "verified"
    state["verified_evidence_cards"] = verified_cards
    state["usable_evidence_cards"] = deepcopy(verified_cards)

    loader, _ = install_llm(monkeypatch, draft(state))
    output = module.synthesis_agent(state)["synthesis_result"]

    assert output["evidence_ids"]
    loader.assert_called_once()
    assert all(
        card["verification_status"] == "unverified"
        for card in state["evidence_cards"]
    )


def test_synthesis_uses_partial_cards_as_provisional_evidence(state, monkeypatch):
    """부분 검증 카드를 사용하되 종합 상태는 자료 부족으로 유지하는지 확인."""
    partial_cards = deepcopy(state["evidence_cards"])
    for card in partial_cards:
        card["verification_status"] = "partially_verified"
    state["verification_result"] = {
        "status": "insufficient_evidence",
        "limitations": ["원문 접근 차단으로 일부 카드만 확인"],
    }
    state["usable_evidence_cards"] = partial_cards

    loader, _ = install_llm(monkeypatch, draft(state))
    output = module.synthesis_agent(state)["synthesis_result"]

    assert output["status"] == "insufficient_evidence"
    assert output["evidence_ids"]
    assert any("부분 검증" in item for item in output["limitations"])
    loader.assert_called_once()


def test_synthesis_rejects_partial_card_as_fact(state, monkeypatch):
    """부분 검증 카드를 사실 문장으로 사용하면 종합을 실패시킨다."""
    partial_cards = deepcopy(state["evidence_cards"])
    for card in partial_cards:
        card["verification_status"] = "partially_verified"
    state["verification_result"] = {"status": "insufficient_evidence"}
    state["usable_evidence_cards"] = partial_cards
    response = draft(state)
    response["summary"][0]["claim_type"] = "fact"
    install_llm(monkeypatch, response)

    output = module.synthesis_agent(state)["synthesis_result"]

    assert output["status"] == "failed"
    assert output["payload"]["diagnostics"][0]["check"] == "partial_card_fact"


def test_synthesis_failure_identifies_failing_node(state, monkeypatch):
    """종합 노드 오류가 발생하면 안전한 실패 결과에 노드명이 포함되는지 확인."""
    verified_state = verified(state)
    response = draft(verified_state)
    response["summary"][0]["evidence_ids"] = ["invented"]
    install_llm(monkeypatch, response)

    output = module.synthesis_agent(verified_state)["synthesis_result"]

    assert output["status"] == "failed"
    assert "_validate_synthesis:ValueError:unknown_evidence_id" in output[
        "errors"
    ][0]
    assert output["payload"]["diagnostics"][0]["evidence_statuses"] == {
        "invented": "missing"
    }


def test_synthesis_accepts_partial_card_as_limitation(state, monkeypatch):
    """부분 검증 근거를 limitation 문장으로 반환하면 정상 종합하는지 확인."""
    partial_cards = deepcopy(state["evidence_cards"])
    for card in partial_cards:
        card["verification_status"] = "partially_verified"
    state["verification_result"] = {
        "status": "insufficient_evidence",
        "limitations": ["원문 접근 차단으로 일부 카드만 확인"],
    }
    state["usable_evidence_cards"] = partial_cards
    response = draft(state)
    response["summary"][0]["claim_type"] = "limitation"
    install_llm(monkeypatch, response)

    output = module.synthesis_agent(state)["synthesis_result"]

    assert output["status"] == "insufficient_evidence"


def test_four_perspectives_success(state, monkeypatch):
    """네 관점 모두에 두 기술의 모의 근거가 있을 때 ok 반환 확인."""
    state = verified(state)
    for p in module.PERSPECTIVES[1:]:
        for original in state["evidence_cards"][:1] + state["evidence_cards"][2:3]:
            card = deepcopy(original)
            card.update(
                evidence_id=f"mock-{p}-{card['technology']}",
                perspective=p,
                source_title="가상 테스트 자료",
                source_url="https://example.test",
            )
            state["evidence_cards"].append(card)
            state["verified_evidence_cards"].append(deepcopy(card))
            state["usable_evidence_cards"].append(deepcopy(card))
    response = draft(state)
    response["comparison_rows"] = [
        {
            "perspective": p,
            "text": "가상 비교",
            "claim_type": "inference",
            "evidence_ids": [
                c["evidence_id"]
                for c in state["evidence_cards"]
                if c["perspective"] == p
            ],
        }
        for p in module.PERSPECTIVES
    ]
    install_llm(monkeypatch, response)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.synthesis_agent(state)["synthesis_result"]["status"] == "ok"


@pytest.mark.parametrize("status", ["partially_verified", "unsupported", "unverified"])
def test_excluded_cards(state, monkeypatch, status):
    """부분 검증, 거절, 미검증의 세 상태 각각에 대해 근거 제외와 LLM 미호출 확인."""
    for card in state["evidence_cards"]:
        card["verification_status"] = status
    loader, _ = install_llm(monkeypatch, {})
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.synthesis_agent(state)["synthesis_result"]["evidence_ids"] == []
    loader.assert_not_called()


def test_unknown_citation_rejected(state, monkeypatch):
    """존재하지 않는 근거 ID를 반환한 LLM 응답의 실패 처리 확인."""
    state = verified(state)
    response = draft(state)
    response["summary"][0]["evidence_ids"] = ["invented"]
    install_llm(monkeypatch, response)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.synthesis_agent(state)["synthesis_result"]["status"] == "failed"


def test_conflicting_duplicate_fails(state, monkeypatch):
    """같은 ID에 다른 주장이 연결되면 LLM 호출 전에 실패하는지 확인."""
    state = verified(state)
    other = deepcopy(state["evidence_cards"][0])
    other["claim"] = "충돌하는 내용"
    state["evidence_cards"].append(other)
    state["verified_evidence_cards"].append(deepcopy(other))
    state["usable_evidence_cards"].append(deepcopy(other))
    loader, _ = install_llm(monkeypatch, {})
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.synthesis_agent(state)["synthesis_result"]["status"] == "failed"
    loader.assert_not_called()


def test_llm_error_redacted(state, monkeypatch):
    """API 오류를 모의하여 실패 상태 반환과 예외 메시지의 비밀정보 노출 방지 확인."""
    _, llm = install_llm(monkeypatch, {})
    # 정상 응답 대신 예외 발생을 모의. SECRET은 노출 검사 전용 가짜 문자열.
    llm.with_structured_output.return_value.invoke.side_effect = RuntimeError("SECRET")
    output = module.synthesis_agent(verified(state))
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert output["synthesis_result"]["status"] == "failed"
    assert "SECRET" not in str(output)


def test_invalid_yaml(state, monkeypatch, tmp_path):
    """임시 YAML 문법 오류가 전체 실행 예외 대신 실패 결과로 반환되는지 확인."""
    path = tmp_path / "bad.yaml"
    path.write_text("system_prompt: [")
    monkeypatch.setattr(module, "PROMPT_PATH", path)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert module.synthesis_agent(state)["synthesis_result"]["status"] == "failed"


def test_fabricated_number_rejected(state, monkeypatch):
    """근거에 없는 98765라는 숫자를 생성한 응답의 차단 확인."""
    state = verified(state)
    reply = draft(state)
    reply["summary"][0]["text"] = "처리량이 98765% 증가"
    install_llm(monkeypatch, reply)
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    output = module.synthesis_agent(state)["synthesis_result"]
    assert output["status"] == "failed"
    assert output["payload"]["diagnostics"][0]["check"] == "fabricated_number"


def test_missing_verification_prevents_llm(state, monkeypatch):
    """카드 판정만 있고 검증 단계 결과가 없으면 종합을 시작하지 않는지 확인."""
    state = verified(state)
    state.pop("verification_result")
    loader, _ = install_llm(monkeypatch, {})
    # 결과 확인: 아래 assert 조건 중 하나라도 다르면 테스트 실패.
    assert (
        module.synthesis_agent(state)["synthesis_result"]["status"]
        == "insufficient_evidence"
    )
    loader.assert_not_called()


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (
            "success",
            ["load_config", "prepare_context", "generate", "validate", "build_result"],
        ),
        ("empty", ["load_config", "prepare_context", "insufficient"]),
        (
            "bad_citation",
            ["load_config", "prepare_context", "generate", "validate", "failure"],
        ),
        ("api_error", ["load_config", "prepare_context", "generate", "failure"]),
        ("bad_config", ["load_config", "failure"]),
    ],
)
def test_synthesis_graph_routes(state, monkeypatch, tmp_path, mode, expected):
    """실제 stream 이벤트로 정상, 자료 부족, 생성 오류, 검증 오류 경로 확인."""
    request = state if mode == "empty" else verified(state)
    before = deepcopy(request)
    reply = draft(request)
    if mode == "bad_citation":
        reply["summary"][0]["evidence_ids"] = ["missing"]
    loader, llm = install_llm(monkeypatch, reply)
    if mode == "api_error":
        llm.with_structured_output.return_value.invoke.side_effect = RuntimeError(
            "SECRET"
        )
    if mode == "bad_config":
        path = tmp_path / "invalid.yaml"
        path.write_text("system_prompt: [")
        monkeypatch.setattr(module, "PROMPT_PATH", path)
    events = list(
        module.build_synthesis_graph().stream(
            {"request": request}, stream_mode="updates"
        )
    )
    assert [name for event in events for name in event] == expected
    output = events[-1][expected[-1]]["output"]
    assert set(output) == {"synthesis_result"}
    assert "SECRET" not in str(events)
    assert request == before
    if mode in ("empty", "bad_config"):
        loader.assert_not_called()
    else:
        llm.with_structured_output.return_value.invoke.assert_called_once()


def test_synthesis_cached_graph_keeps_runs_separate(state, monkeypatch):
    """동일한 컴파일 그래프를 재사용해도 이전 카드와 오류가 다음 실행에 남지 않는지 확인."""
    request = verified(state)
    _, llm = install_llm(monkeypatch, draft(request))
    first = module.synthesis_agent(request)
    assert first["synthesis_result"]["evidence_ids"]
    second = module.synthesis_agent(state)
    assert second["synthesis_result"]["evidence_ids"] == []
    llm.with_structured_output.return_value.invoke.assert_called_once()
