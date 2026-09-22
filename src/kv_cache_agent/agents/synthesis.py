"""평가 종합 에이전트: 검증된 근거의 비교와 조건부 판단.

인풋:
    GlobalState의 user_query, technical_result, market_result,
    stakeholder_result, cloud_domain_result, verification_result, evidence_cards.
    작성 지침은 prompts/synthesis.yaml에서 로드.
함수 기능:
    synthesis_agent: 입력 정리 → 검증 카드 선별 → LLM 종합 → 근거 검사.
    _verified_cards: 명시적 verified 판정과 출처를 갖춘 카드만 선별.
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
        evidence_cards: list[EvidenceCard]
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
                     "claim_type": "fact" | "inference"}
    Comparison 형식: Statement 필드 + "perspective" 필드.
    perspective 값: technical | market | stakeholder | cloud_domain.
    근거 부족으로 생성 생략 또는 처리 실패 시 payload는 빈 딕셔너리.
"""

import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from langchain_core.messages import HumanMessage, SystemMessage
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
    claim_type: Literal["fact", "inference"]


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
    """카드의 명시적 판정만 사용. 검증 노드의 ok로 카드를 승격하지 않음."""
    cards: dict[str, dict] = {}
    for card in state.get("evidence_cards", []):
        key = card.get("evidence_id")
        if not isinstance(key, str) or not key.strip():
            raise ValueError("Missing evidence ID")
        if key in cards and cards[key] != card:
            raise ValueError("Conflicting evidence ID")
        cards[key] = deepcopy(card)
    if state.get("verification_result", {}).get("status") != "ok":
        return {}, ["근거 검증 완료 상태 미확보"]
    allowed = {
        key: card
        for key, card in cards.items()
        if card.get("verification_status") == "verified"
        and all(
            isinstance(card.get(k), str) and card[k].strip()
            for k in ("claim", "evidence_text", "source_title", "source_url")
        )
    }
    limitations = []
    if len(allowed) != len(cards):
        limitations.append("미검증, 부분 검증, 거절 또는 출처 불완전 근거 제외")
    return allowed, limitations


def _validate_statement(statement: Statement, cards: dict[str, dict]) -> None:
    """생성 주장과 허용 카드 입력 → ID, 빈 문장, 숫자 검사 → 실패 시 예외."""
    if not set(statement.evidence_ids) <= cards.keys():
        raise ValueError("Unknown evidence ID")
    if not statement.text.strip():
        raise ValueError("Empty statement")
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


def synthesis_agent(state: GlobalState) -> dict[str, Any]:
    """새 검색 없이 검증 근거를 비교하고 담당 State 갱신값만 반환."""
    try:
        config = _load_config(PROMPT_PATH)
        cards, limitations = _verified_cards(state)
        limitations.extend(state.get("verification_result", {}).get("limitations", []))
        for perspective in PERSPECTIVES:
            limitations.extend(
                state.get(f"{perspective}_result", {}).get("limitations", [])
            )
            if not any(c.get("perspective") == perspective for c in cards.values()):
                limitations.append(f"{perspective}: 검증 근거 부족")
        if not cards:
            return _result(
                "insufficient_evidence",
                "검증 완료 근거 부족으로 판단 보류",
                limitations=limitations,
            )
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
                p: state.get(f"{p}_result", {}).get("limitations", [])
                for p in PERSPECTIVES
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
        draft = SynthesisDraft.model_validate(response)
        statements = [
            *draft.summary,
            *draft.comparison_rows,
            *draft.agreements,
            *draft.conflicts,
            *draft.conditional_recommendations,
        ]
        for statement in statements:
            _validate_statement(statement, cards)
        for row in draft.comparison_rows:
            if any(
                cards[i].get("perspective") != row.perspective for i in row.evidence_ids
            ):
                raise ValueError("Comparison perspective mismatch")
        covered = {row.perspective for row in draft.comparison_rows}
        for p in PERSPECTIVES:
            if p not in covered:
                limitations.append(f"{p}: 비교 결과 미작성")
        complete = all(
            any(
                row.perspective == p
                and any(
                    cards[i].get("technology") in (tech, "both")
                    for i in row.evidence_ids
                )
                for row in draft.comparison_rows
            )
            for p in PERSPECTIVES
            for tech in ("TurboQuant", "CXL-based")
        )
        if not complete:
            limitations.append("네 관점에서 두 기술의 비교 근거가 모두 확보되지 않음")
        ids = list(dict.fromkeys(i for item in statements for i in item.evidence_ids))
        return _result(
            "ok" if complete else "insufficient_evidence",
            "\n".join(item.text for item in draft.summary),
            ids,
            list(dict.fromkeys([*limitations, *draft.limitations])),
            payload=draft.model_dump(),
        )
    except Exception as error:  # noqa: BLE001 - 노드 경계에서 실패 결과로 변환
        # 외부 예외 메시지에는 키 또는 요청 본문이 포함될 수 있으므로 타입만 기록.
        return _result(
            "failed",
            "평가 종합 실패",
            errors=[f"종합 처리 오류 ({type(error).__name__})"],
        )
