# 내부 LangGraph: 순차 처리와 조건부 분기로 구성, 반복 루프 없음.
# 정상: START → load_config → prepare_context → generate → validate → build_result → END
# 근거 없음: prepare_context → insufficient → END (LLM 호출 생략)
# 처리 오류: 해당 노드 → failure → END
# 외부 반환: {"synthesis_result": AgentResult}, 내부 상태: SynthesisState

"""평가 종합 에이전트: 검증 근거의 비교와 조건부 판단.

인풋:
    GlobalState의 user_query, technical_result, market_result,
    stakeholder_result, cloud_domain_result, verification_result,
    evidence_cards, verified_evidence_cards, usable_evidence_cards.
    작성 지침은 prompts/synthesis.yaml에서 로드.
함수 기능:
    synthesis_agent: 내부 LangGraph 호출 후 기존 반환 형식으로 결과 전달.
    build_synthesis_graph: 설정 → 선별 → 생성 → 검증 → 결과 조립 노드 연결.
    조건부 경로: 근거 부족은 insufficient, 처리 오류는 failure 노드로 이동.
    _verified_cards: verified 또는 partially_verified 카드를 선별.
    _validate_statement / _validate_numbers: 근거 ID와 숫자의 출처 존재 확인.
    _result: 기존 AgentResult 필드에 결과, 한계, 오류 구성.
아웃풋:
    {"synthesis_result": AgentResult} 형태의 State 갱신값.
    근거 부족은 insufficient_evidence, 처리 오류는 failed로 반환.
    입력 State 변경, 신규 검색, 다른 에이전트 직접 호출 없음.

검증 범위:
    숫자와 인용 ID의 존재 검사이며, 문장의 의미적 타당성 전체를 보장하지 않음.

입출력 형식:
    함수: synthesis_agent(state: GlobalState) -> dict[str, Any]
    입력 필드:
        user_query: str
        technical_result, market_result, stakeholder_result,
        cloud_domain_result, verification_result: AgentResult
        evidence_cards, verified_evidence_cards, usable_evidence_cards: list[EvidenceCard]
    GlobalState와 AgentResult는 TypedDict(total=False)로 정의된 딕셔너리.
    아래는 타입 설명용 표기이며 실제 값은 해당 자료형의 데이터로 전달.

    반환 구조:
        {
            "synthesis_result": {
                "agent_name": str,
                "status": "ok" | "insufficient_evidence" | "failed",
                "summary": str,
                "evidence_ids": list[str],
                "limitations": list[str],
                "errors": list[str],
                "payload": dict[str, Any],
            }
        }

    공통 명세의 needs_retry도 유효한 상태이나 현재 함수에서는 생성하지 않음.

    성공적으로 생성된 payload의 내부 구조:
        summary: list[Statement]
        comparison_rows: list[Comparison]
        agreements, conflicts, conditional_recommendations: list[Statement]
        limitations: list[str]

    실제 payload에는 모델 객체가 아닌 model_dump() 결과인 dict와 list 저장.
    Statement 형식: {"text": str, "evidence_ids": list[str],
                     "claim_type": "fact" | "inference" | "limitation"}
    Comparison 형식: Statement 필드 + "perspective" 필드.
    perspective 값: technical | market | stakeholder | cloud_domain.
    근거 부족으로 생성 생략 또는 처리 실패 시 payload는 빈 딕셔너리.
"""

import json
import re
from copy import deepcopy
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "synthesis.yaml"
PERSPECTIVES = ("technical", "market", "stakeholder", "cloud_domain")


class Statement(BaseModel):
    """노드 내부 생성 형식. 공통 State 스키마와는 별개."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    claim_type: Literal["fact", "inference", "limitation"]


class Comparison(Statement):
    """관점 구분과 근거 ID를 갖는 내부 비교 항목."""

    perspective: Literal["technical", "market", "stakeholder", "cloud_domain"]


class SynthesisDraft(BaseModel):
    """LLM 종합 응답 검사용 내부 모델이며 공통 State 형식은 유지."""

    model_config = ConfigDict(extra="forbid")
    summary: list[Statement] = Field(min_length=1)
    comparison_rows: list[Comparison] = Field(min_length=1)
    agreements: list[Statement]
    conflicts: list[Statement]
    conditional_recommendations: list[Statement]
    limitations: list[str]


def _load_config(path: Path) -> dict[str, Any]:
    """YAML 경로 입력 → 필수 프롬프트와 검색 금지 설정 검사 → 설정 딕셔너리 반환."""
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get("system_prompt"), str):
        raise TypeError("Invalid prompt configuration")
    if not config["system_prompt"].strip():
        raise ValueError("Empty prompt")
    if config.get("constraints", {}).get("allow_new_search") is not False:
        raise ValueError("Search must be disabled")
    return config


def _verified_cards(state: GlobalState) -> tuple[dict[str, dict], list[str]]:
    """GlobalState의 검증·부분 검증 카드를 잠정 종합 입력으로 선별한다."""
    cards: dict[str, dict] = {}
    for card in state.get("usable_evidence_cards", []):
        key = card.get("evidence_id")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Missing evidence ID")
        if key in cards and cards[key] != card:
            raise ValueError("Conflicting evidence ID")
        cards[key] = deepcopy(card)
    verification_status = state.get("verification_result", {}).get("status")
    if verification_status not in {"ok", "insufficient_evidence"}:
        return {}, ["사용 가능한 근거 검증 결과 미확보"]
    allowed = {
        key: card
        for key, card in cards.items()
        if card.get("verification_status")
        in {"verified", "partially_verified"}
        and all(
            isinstance(card.get(k), str) and card[k].strip()
            for k in ("claim", "evidence_text", "source_title", "source_url")
        )
    }
    limitations = []
    if any(
        card.get("verification_status") == "partially_verified"
        for card in allowed.values()
    ):
        limitations.append(
            "부분 검증 근거를 포함한 잠정 평가이며 확정적 사실로 해석하면 안 됩니다."
        )
    if len(allowed) != len(cards):
        limitations.append("미검증, 거절 또는 출처 불완전 근거는 제외했습니다.")
    return allowed, limitations


def _validate_statement(statement: Statement, cards: dict[str, dict]) -> None:
    """생성 주장과 허용 카드 입력 → ID, 유형, 빈 문장, 숫자 검사 → 실패 시 예외."""
    if not set(statement.evidence_ids) <= cards.keys():
        raise ValueError("Unknown evidence ID")
    if not statement.text.strip():
        raise ValueError("Empty statement")
    if statement.claim_type == "fact" and any(
        cards[evidence_id].get("verification_status") == "partially_verified"
        for evidence_id in statement.evidence_ids
    ):
        raise ValueError("Partially verified evidence cannot support fact")
    _validate_numbers(statement.text, statement.evidence_ids, cards)


def _validate_numbers(text: str, ids: list[str], cards: dict[str, dict]) -> None:
    """새 숫자 생성을 차단. 같은 숫자의 의미와 실험 조건까지 입증하지는 않음."""
    pattern = r"(?<!\d)\d+(?:[.,]\d+)*"
    source = " ".join(
        str(cards[key].get(field, ""))
        for key in ids
        for field in ("claim", "evidence_text", "caveat")
    )
    if not set(re.findall(pattern, text)) <= set(re.findall(pattern, source)):
        raise ValueError("Numeric claim absent from cited evidence")


def _validation_error_code(error: ValueError) -> str:
    """검증 예외를 로그용 비민감 오류 코드로 변환한다."""
    labels = (
        ("Unknown evidence ID", "unknown_evidence_id"),
        ("Empty statement", "empty_statement"),
        (
            "Partially verified evidence cannot support fact",
            "partial_card_fact",
        ),
        ("Numeric claim absent from cited evidence", "fabricated_number"),
        ("Comparison perspective mismatch", "perspective_mismatch"),
    )
    message = str(error)
    for fragment, label in labels:
        if fragment in message:
            return label
    return "validation_error"


def _statement_diagnostic(
    error: ValueError,
    group: str,
    index: int,
    statement: Statement,
    cards: dict[str, dict],
) -> dict[str, Any]:
    """주장 검증 실패를 재현 가능한 최소 정보로 기록한다."""
    evidence_ids = list(statement.evidence_ids)
    return {
        "node": "_validate_synthesis",
        "check": _validation_error_code(error),
        "statement_group": group,
        "statement_index": index,
        "claim_type": statement.claim_type,
        "evidence_ids": evidence_ids,
        "evidence_statuses": {
            evidence_id: cards.get(evidence_id, {}).get(
                "verification_status", "missing"
            )
            for evidence_id in evidence_ids
        },
        "text_preview": statement.text[:200],
    }


def _perspective_diagnostic(
    row_index: int, row: Comparison, cards: dict[str, dict]
) -> dict[str, Any]:
    """비교 관점과 카드 관점 불일치 정보를 기록한다."""
    return {
        "node": "_validate_synthesis",
        "check": "perspective_mismatch",
        "statement_group": "comparison_rows",
        "statement_index": row_index,
        "perspective": row.perspective,
        "evidence_ids": list(row.evidence_ids),
        "evidence_perspectives": {
            evidence_id: cards.get(evidence_id, {}).get(
                "perspective", "missing"
            )
            for evidence_id in row.evidence_ids
        },
        "text_preview": row.text[:200],
    }


def _result(
    status: str, summary: str, ids=None, limitations=None, errors=None, payload=None
):
    """상태와 생성 내용 입력 → 기존 필드 구성 → synthesis_result 갱신값 반환."""
    return {
        "synthesis_result": {
            "agent_name": "synthesis",
            "status": status,
            "summary": summary,
            "evidence_ids": ids or [],
            "limitations": limitations or [],
            "errors": errors or [],
            "payload": payload or {},
        }
    }


class SynthesisState(TypedDict, total=False):
    """서브그래프 전용 상태. GlobalState에는 이 필드를 추가하지 않음.

    request: 외부 입력 사본, config: YAML 설정, cards: 허용 근거 ID 매핑.
    response: LLM 원시 응답, draft: 검사된 내부 모델의 딕셔너리.
    output: 기존 외부 반환값, error_type: 비밀정보 없는 실패 종류.
    """

    request: GlobalState
    config: dict[str, Any]
    cards: dict[str, dict]
    limitations: list[str]
    response: Any
    draft: dict[str, Any]
    complete: bool
    evidence_ids: list[str]
    error_type: str
    diagnostics: list[dict[str, Any]]
    output: dict[str, Any]


def _guard_node(node):
    """노드 예외를 오류 상태로 변환하여 그래프의 실패 경로로 전달."""

    @wraps(node)
    def guarded(state):
        try:
            return node(state)
        except Exception as error:  # noqa: BLE001 - 노드 경계의 안전한 오류 변환
            return {
                "error_type": f"{node.__name__}:{type(error).__name__}",
                "diagnostics": [
                    {
                        "node": node.__name__,
                        "check": "unhandled_exception",
                        "error_type": type(error).__name__,
                    }
                ],
            }

    return guarded


def _route_error(state) -> str:
    """오류 상태의 유무에 따라 다음 노드 또는 실패 노드 선택."""
    return "error" if state.get("error_type") else "next"


@_guard_node
def _load_synthesis_config(state: SynthesisState) -> dict:
    """입력: 내부 상태 → 처리: YAML 로드 → 출력: config."""
    return {"config": _load_config(PROMPT_PATH)}


@_guard_node
def _prepare_context(local: SynthesisState) -> dict:
    """입력: request → 처리: 근거 선별과 누락 관점 확인 → 출력: cards, limitations."""
    state = local["request"]
    cards, limitations = _verified_cards(state)
    limitations.extend(state.get("verification_result", {}).get("limitations", []))
    for perspective in PERSPECTIVES:
        limitations.extend(
            state.get(f"{perspective}_result", {}).get("limitations", [])
        )
        if not any(c.get("perspective") == perspective for c in cards.values()):
            limitations.append(f"{perspective}: 검증 근거 부족")
    return {"cards": cards, "limitations": limitations}


def _route_evidence(state: SynthesisState) -> str:
    """근거가 없으면 LLM 노드를 건너뛰고 자료 부족 노드로 이동."""
    if state.get("error_type"):
        return "error"
    return "ready" if state["cards"] else "empty"


@_guard_node
def _generate_synthesis(local: SynthesisState) -> dict:
    """입력: request, cards, config → 처리: LLM 1회 호출 → 출력: response."""
    state, cards, config = local["request"], local["cards"], local["config"]
    context = {
        "user_query": state.get("user_query", ""),
        "evidence_cards": list(cards.values()),
        "perspective_results": {
            p: {
                "status": state.get(f"{p}_result", {}).get("status"),
                "summary": state.get(f"{p}_result", {}).get("summary", ""),
                "evidence_ids": [
                    key
                    for key in state.get(f"{p}_result", {}).get("evidence_ids", [])
                    if key in cards
                ],
            }
            for p in PERSPECTIVES
            if any(
                key in cards
                for key in state.get(f"{p}_result", {}).get("evidence_ids", [])
            )
        },
        "perspective_limitations": {
            p: state.get(f"{p}_result", {}).get("limitations", []) for p in PERSPECTIVES
        },
    }
    response = (
        get_llm()
        .with_structured_output(SynthesisDraft)
        .invoke(
            [
                SystemMessage(content=config["system_prompt"]),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    )
    return {"response": response}


@_guard_node
def _validate_synthesis(local: SynthesisState) -> dict:
    """입력: response, cards → 처리: 응답과 근거 검사 → 출력: draft와 완료 여부."""
    draft = SynthesisDraft.model_validate(local["response"])
    cards = local["cards"]
    limitations = list(local["limitations"])
    statement_groups = (
        ("summary", draft.summary),
        ("comparison_rows", draft.comparison_rows),
        ("agreements", draft.agreements),
        ("conflicts", draft.conflicts),
        ("conditional_recommendations", draft.conditional_recommendations),
    )
    statements = []
    for group_name, group in statement_groups:
        for index, statement in enumerate(group):
            try:
                _validate_statement(statement, cards)
            except ValueError as error:
                return {
                    "error_type": (
                        "_validate_synthesis:ValueError:"
                        f"{_validation_error_code(error)}"
                    ),
                    "diagnostics": [
                        _statement_diagnostic(
                            error, group_name, index, statement, cards
                        )
                    ],
                }
            statements.append(statement)
    for row_index, row in enumerate(draft.comparison_rows):
        if any(
            cards[i].get("perspective") != row.perspective for i in row.evidence_ids
        ):
            return {
                "error_type": "_validate_synthesis:ValueError:perspective_mismatch",
                "diagnostics": [
                    _perspective_diagnostic(row_index, row, cards)
                ],
            }
    covered = {row.perspective for row in draft.comparison_rows}
    for p in PERSPECTIVES:
        if p not in covered:
            limitations.append(f"{p}: 비교 결과 미작성")
    complete = all(
        any(
            row.perspective == p
            and any(
                cards[i].get("technology") in (tech, "both") for i in row.evidence_ids
            )
            for row in draft.comparison_rows
        )
        for p in PERSPECTIVES
        for tech in ("TurboQuant", "CXL-based")
    )
    if not complete:
        limitations.append("네 관점에서 두 기술의 비교 근거가 모두 확보되지 않음")
    ids = list(dict.fromkeys(i for item in statements for i in item.evidence_ids))
    return {
        "draft": draft.model_dump(),
        "complete": complete,
        "evidence_ids": ids,
        "limitations": list(dict.fromkeys([*limitations, *draft.limitations])),
    }


@_guard_node
def _build_result(local: SynthesisState) -> dict:
    """입력: 검사된 draft → 처리: 기존 AgentResult 조립 → 출력: output."""
    draft = local["draft"]
    return {
        "output": _result(
            "ok" if local["complete"] else "insufficient_evidence",
            "\n".join(item["text"] for item in draft["summary"]),
            local["evidence_ids"],
            local["limitations"],
            payload=draft,
        )
    }


def _insufficient_result(local: SynthesisState) -> dict:
    """근거가 없는 경로의 종료 결과. LLM 호출 없음."""
    return {
        "output": _result(
            "insufficient_evidence",
            "검증 완료 근거 부족으로 판단 보류",
            limitations=local["limitations"],
        )
    }


def _synthesis_failure(local: SynthesisState) -> dict:
    """오류 노드의 종료 결과. 원본 예외 메시지는 외부로 전달하지 않음."""
    diagnostics = local.get("diagnostics", [])
    return {
        "output": _result(
            "failed",
            "평가 종합 실패",
            errors=[f"종합 처리 오류 ({local['error_type']})"],
            payload={"diagnostics": diagnostics} if diagnostics else {},
        )
    }


@lru_cache(maxsize=1)
def build_synthesis_graph():
    """내부 StateGraph 컴파일. 실행 상태는 invoke마다 분리, 체크포인터 없음."""
    graph = StateGraph(SynthesisState)
    graph.add_node("load_config", _load_synthesis_config)
    graph.add_node("prepare_context", _prepare_context)
    graph.add_node("generate", _generate_synthesis)
    graph.add_node("validate", _validate_synthesis)
    graph.add_node("build_result", _build_result)
    graph.add_node("insufficient", _insufficient_result)
    graph.add_node("failure", _synthesis_failure)
    graph.add_edge(START, "load_config")
    graph.add_conditional_edges(
        "load_config", _route_error, {"next": "prepare_context", "error": "failure"}
    )
    graph.add_conditional_edges(
        "prepare_context",
        _route_evidence,
        {"ready": "generate", "empty": "insufficient", "error": "failure"},
    )
    for source, target in (
        ("generate", "validate"),
        ("validate", "build_result"),
        ("build_result", END),
    ):
        graph.add_conditional_edges(
            source, _route_error, {"next": target, "error": "failure"}
        )
    graph.add_edge("insufficient", END)
    graph.add_edge("failure", END)
    return graph.compile()


def synthesis_agent(state: GlobalState) -> dict[str, Any]:
    """GlobalState를 내부 그래프에 전달하고 synthesis_result만 외부에 반환."""
    try:
        result = build_synthesis_graph().invoke({"request": deepcopy(state)})
        return result["output"]
    except Exception as error:  # noqa: BLE001 - 그래프 실행 경계의 오류 처리
        return _synthesis_failure(
            {
                "error_type": f"synthesis_agent:{type(error).__name__}",
                "diagnostics": [
                    {
                        "node": "synthesis_agent",
                        "check": "unhandled_exception",
                        "error_type": type(error).__name__,
                    }
                ],
            }
        )["output"]
