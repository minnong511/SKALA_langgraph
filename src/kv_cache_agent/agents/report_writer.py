# 내부 LangGraph: 순차 처리와 조건부 분기로 구성, 반복 루프 없음.
# 정상: START → load_config → prepare_context → generate → validate_sections
#       → render → validate_report → END
# 자료 부족: prepare_context → fallback → validate_report → END (LLM 호출 생략)
# 처리 오류 또는 종합 실패: 해당 노드 → failure → END
# 외부 반환: {"final_report": str}, 내부 상태: ReportState

"""보고서 생성 에이전트: 종합 결과의 문서화와 출처 연결.

인풋:
    GlobalState의 user_query, synthesis_result, verification_result,
    evidence_cards, 선택 입력 research_plan.
    목차와 절별 지침은 prompts/report_writer.yaml에서 로드.

함수 기능:
    report_writer_agent: 내부 LangGraph 호출 후 기존 문자열 형식으로 결과 전달.
    build_report_graph: 설정 → 입력 → 생성 → 인용 검사 → 조립 → 최종 검사 연결.
    조건부 경로: 자료 부족은 fallback, 처리 오류는 failure 노드로 이동.
    _load_prompt_config: 최상위 목차 8개와 하위 절 20개의 구조 검사.
    _render_markdown: 고정 제목, 인용 번호, 실제 사용한 참고문헌 조립.
    _fallback: 검증 자료 부족 시 외부 호출 없이 안내 보고서 구성.

아웃풋:
    {"final_report": str} 형태의 Markdown 문자열 갱신값.
    생성 실패 시에도 같은 문자열 형식으로 안전한 실패 안내 반환.
    관점별 원본 평가의 재평가, 신규 검색, PDF 생성과 파일 저장 없음.

검증 범위:
    SUMMARY의 PDF 반 페이지 조건은 별도 PDF 렌더링 단계에서 확인 필요.

입출력 형식:
    함수: report_writer_agent(state: GlobalState) -> dict[str, Any]
    입력 필드:
        user_query: str
        synthesis_result, verification_result: AgentResult
        evidence_cards, verified_evidence_cards: list[EvidenceCard]
        research_plan: ResearchPlan  # 선택 입력
    반환 구조: {"final_report": str}
    정상 문자열 구조: SUMMARY → 본문 6개 장과 하위 절 20개 → REFERENCE.
    실패 문자열 구조: 보고서 생성 실패 제목과 안전한 오류 안내.
    반환값은 PDF 파일 경로나 AgentResult 객체가 아닌 Markdown 문자열.

LLM 내부 응답 형식:
    {
        "sections": [
            {
                "section_id": str,
                "paragraphs": [
                    {
                        "text": str,
                        "claim_type": "fact" | "inference" | "limitation",
                        "evidence_ids": list[str],
                    }
                ],
            }
        ]
    }
    위 구조는 타입 설명용 표기이며 실제 응답은 각 자료형의 값으로 구성.
    section_id는 YAML의 summary와 하위 절 ID 20개로 구성.
    fact와 inference는 근거 ID 필수, limitation은 빈 ID 목록 허용.
    내부 sections 구조는 외부 State에 반환하지 않고 문자열로 조립 후 반환.
"""

import json
import re
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field

from kv_cache_agent.agents.synthesis import (
    _guard_node,
    _load_config,
    _route_error,
    _validate_numbers,
    _verified_cards,
)
from kv_cache_agent.graph.state import GlobalState
from kv_cache_agent.llm import get_llm

PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "report_writer.yaml"
TITLES = [
    "SUMMARY",
    "1. 분석 배경",
    "2. 기술 선정",
    "3. 기술 개요",
    "4. 관점 별 평가",
    "5. 시사점",
    "6. 한계점",
    "REFERENCE",
]


class Paragraph(BaseModel):
    """사실, 추론, 한계와 관련 근거 ID를 담는 본문 문단."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    claim_type: Literal["fact", "inference", "limitation"]
    evidence_ids: list[str]


class Section(BaseModel):
    """YAML의 절 ID에 대응하는 생성 문단 목록."""

    model_config = ConfigDict(extra="forbid")
    section_id: str
    paragraphs: list[Paragraph] = Field(min_length=1)


class ReportDraft(BaseModel):
    """SUMMARY와 하위 절의 생성 응답 검사용 내부 모델."""

    model_config = ConfigDict(extra="forbid")
    sections: list[Section]


def _load_prompt_config() -> dict:
    """보고서 YAML 입력 → 제목 순서와 절 ID 검사 → 설정 딕셔너리 반환."""
    config = _load_config(PROMPT_PATH)
    structure = config.get("report_structure", [])
    if [s["title"] for s in structure] != TITLES:
        raise ValueError("Invalid report outline")
    ids = []
    for index, section in enumerate(structure):
        ids.append(section["id"])
        children = section.get("subsections", [])
        if index in range(1, 7):
            count = (3, 3, 3, 4, 3, 4)[index - 1]
            if len(children) != count or any(
                not child["title"].startswith(f"{index}.{number} ")
                for number, child in enumerate(children, 1)
            ):
                raise ValueError("Invalid subsection order")
        for child in children:
            ids.append(child["id"])
            if not child.get("instruction"):
                raise ValueError("Missing instruction")
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate section ID")
    return config


def _body_sections(config: dict) -> list[dict]:
    """목차 설정 입력 → SUMMARY와 하위 절만 추출 → 본문 구역 목록 반환."""
    return [
        child
        for section in config["report_structure"][:-1]
        for child in section.get("subsections", [section])
    ]


def _plain(text: str) -> str:
    # LLM 본문이나 외부 메타데이터로 제목, 링크, 인용 번호를 삽입하지 않도록 처리.
    """본문 또는 메타데이터 입력 → 구조용 문자와 줄바꿈 정리 → 표시 문자열 반환."""
    return re.sub(r"[\[\]#<>`*]", "", " ".join(text.split()))


def _render_markdown(
    config: dict, sections: dict[str, list[Paragraph]], cards: dict
) -> str:
    """절별 본문과 근거 입력 → 고정 목차와 번호별 출처 조립 → Markdown 반환."""
    numbers: dict[str, int] = {}
    lines = []
    for section in config["report_structure"][:-1]:
        lines.extend([f"# {section['title']}", ""])
        for child in section.get("subsections", [section]):
            if child is not section:
                lines.extend([f"## {child['title']}", ""])
            for paragraph in sections[child["id"]]:
                for key in paragraph.evidence_ids:
                    if key not in numbers:
                        numbers[key] = len(numbers) + 1
                citations = " ".join(
                    f"[{numbers[i]}]" for i in dict.fromkeys(paragraph.evidence_ids)
                )
                prefix = {"fact": "", "inference": "해석: ", "limitation": "한계: "}[
                    paragraph.claim_type
                ]
                lines.extend(
                    [f"{prefix}{_plain(paragraph.text)} {citations}".strip(), ""]
                )
    lines.extend(["# REFERENCE", ""])
    for key, number in numbers.items():
        card = cards[key]
        # 공통 카드 스키마에 저자, 학회, 조회일 필드가 없으므로 임의 보완하지 않음.
        metadata = ". ".join(
            _plain(str(card.get(k) or default))
            for k, default in (
                ("published_date", "발행일 미상"),
                ("source_title", "제목 미상"),
                ("source_locator", "위치 미제공"),
                ("source_url", "출처 미제공"),
            )
        )
        lines.extend([f"[{number}] {metadata}. 근거 ID: {_plain(key)}", ""])
    if not numbers:
        lines.append("본문에 사용한 검증 근거 없음")
    return "\n".join(lines).strip() + "\n"


def _fallback(config: dict, reason: str, limitations: list[str]) -> str:
    """부족 사유와 한계 입력 → 지정 목차별 안내 구성 → Markdown 반환."""
    text = " / ".join([reason, *limitations])
    sections = {
        child["id"]: [
            Paragraph(
                text=text
                if child["id"] == "summary"
                else "검증 자료 부족으로 해당 항목 판단 보류",
                claim_type="limitation",
                evidence_ids=[],
            )
        ]
        for child in _body_sections(config)
    }
    return _render_markdown(config, sections, {})


class ReportState(TypedDict, total=False):
    """보고서 서브그래프 전용 상태. final_report 외의 내부 값은 외부 반환 금지."""

    request: GlobalState
    config: dict[str, Any]
    result: dict[str, Any]
    cards: dict[str, dict]
    limitations: list[str]
    fallback_reason: str
    upstream_failed: bool
    response: Any
    sections: dict[str, list[Paragraph]]
    markdown: str
    error_type: str
    output: dict[str, str]


@_guard_node
def _load_report_config(local: ReportState) -> dict:
    """입력: 내부 상태 → 처리: YAML과 목차 검사 → 출력: config."""
    return {"config": _load_prompt_config()}


@_guard_node
def _prepare_report_context(local: ReportState) -> dict:
    """입력: request → 처리: 종합 상태와 허용 근거 확인 → 출력: 생성 또는 안내 자료."""
    state = local["request"]
    result = state.get("synthesis_result", {})
    if result.get("status") == "failed":
        return {"upstream_failed": True}
    if result.get("status") not in ("ok", "insufficient_evidence"):
        return {"fallback_reason": "평가 종합 결과 미확보", "limitations": []}
    allowed, limitations = _verified_cards(state)
    ids = result.get("evidence_ids", [])
    if not isinstance(ids, list) or not set(ids) <= allowed.keys():
        raise ValueError("Invalid synthesis evidence IDs")
    cards = {key: allowed[key] for key in ids}
    if not cards:
        return {
            "fallback_reason": "검증 완료 근거 부족으로 판단 보류",
            "limitations": [*limitations, *result.get("limitations", [])],
        }
    return {"cards": cards, "result": result}


def _route_report_input(local: ReportState) -> str:
    """오류와 상류 실패는 failure, 자료 부족은 fallback, 정상 입력은 generate로 이동."""
    if local.get("error_type") or local.get("upstream_failed"):
        return "error"
    return "fallback" if local.get("fallback_reason") else "ready"


@_guard_node
def _generate_sections(local: ReportState) -> dict:
    """입력: 종합과 근거, YAML → 처리: LLM 1회 호출 → 출력: response."""
    state, cards, config = local["request"], local["cards"], local["config"]
    result = local["result"]
    context = {
        "user_query": state.get("user_query", ""),
        "synthesis_result": result,
        "evidence_cards": list(cards.values()),
        "verification_result": state.get("verification_result", {}),
        "research_plan": state.get("research_plan", {}),
        "report_structure": config["report_structure"],
        "reference_formats": config.get("reference_formats", {}),
        "summary_layout_target": config.get("summary_layout_target", {}),
    }
    response = (
        get_llm()
        .with_structured_output(ReportDraft)
        .invoke(
            [
                SystemMessage(content=config["system_prompt"]),
                HumanMessage(content=json.dumps(context, ensure_ascii=False)),
            ]
        )
    )
    return {"response": response}


@_guard_node
def _validate_sections(local: ReportState) -> dict:
    """입력: response → 처리: 절 ID, 인용, 수치 검사 → 출력: sections."""
    response, cards = local["response"], local["cards"]
    result = local["result"]
    outline = _body_sections(local["config"])
    draft = ReportDraft.model_validate(response)
    expected = [s["id"] for s in outline]
    actual = [s.section_id for s in draft.sections]
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("Missing or duplicate section")
    for section in draft.sections:
        for paragraph in section.paragraphs:
            if not set(paragraph.evidence_ids) <= cards.keys():
                raise ValueError("Unknown citation")
            if paragraph.claim_type != "limitation" and not paragraph.evidence_ids:
                raise ValueError("Uncited assertion")
            if paragraph.claim_type != "limitation":
                _validate_numbers(paragraph.text, paragraph.evidence_ids, cards)
            if not paragraph.text.strip() or re.search(
                r"\[[^]]+\]|^\s*#", paragraph.text
            ):
                raise ValueError("Inline citation or heading is not permitted")
    sections = {s.section_id: s.paragraphs for s in draft.sections}
    # 상류에서 확인한 자료 부족은 LLM의 누락 여부와 무관하게 보고서에 보존.
    if result.get("limitations"):
        sections["section_6_1"].append(
            Paragraph(
                text=" / ".join(result["limitations"]),
                claim_type="limitation",
                evidence_ids=[],
            )
        )
    return {"sections": sections}


@_guard_node
def _render_report(local: ReportState) -> dict:
    """입력: 검사된 절과 근거 → 처리: 제목과 참고문헌 조립 → 출력: markdown."""
    return {
        "markdown": _render_markdown(local["config"], local["sections"], local["cards"])
    }


@_guard_node
def _validate_report(local: ReportState) -> dict:
    """입력: markdown → 처리: 최종 제목 순서 검사 → 출력: 기존 final_report 형식."""
    markdown = local["markdown"]
    if re.findall(r"^# (.+)$", markdown, re.MULTILINE) != TITLES:
        raise ValueError("Rendered outline mismatch")
    return {"output": {"final_report": markdown}}


@_guard_node
def _build_fallback_report(local: ReportState) -> dict:
    """자료 부족 안내도 지정 목차로 구성한 뒤 최종 검사 노드로 전달."""
    return {
        "markdown": _fallback(
            local["config"], local["fallback_reason"], local["limitations"]
        )
    }


def _report_failure(local: ReportState) -> dict:
    """오류 경로에서도 문자열 계약 유지. 예외 메시지와 내부 상태 노출 금지."""
    if local.get("upstream_failed"):
        reason = "평가 종합 실패로 작성 중단"
    else:
        reason = f"보고서 처리 오류 ({local['error_type']})"
    return {"output": {"final_report": f"# 보고서 생성 실패\n{reason}\n"}}


@lru_cache(maxsize=1)
def build_report_graph():
    """생성, 인용 검사, 렌더링, 최종 검사를 분리한 LangGraph 구성."""
    graph = StateGraph(ReportState)
    graph.add_node("load_config", _load_report_config)
    graph.add_node("prepare_context", _prepare_report_context)
    graph.add_node("generate", _generate_sections)
    graph.add_node("validate_sections", _validate_sections)
    graph.add_node("render", _render_report)
    graph.add_node("validate_report", _validate_report)
    graph.add_node("fallback", _build_fallback_report)
    graph.add_node("failure", _report_failure)
    graph.add_edge(START, "load_config")
    graph.add_conditional_edges(
        "prepare_context",
        _route_report_input,
        {"ready": "generate", "fallback": "fallback", "error": "failure"},
    )
    for source, target in (
        ("load_config", "prepare_context"),
        ("generate", "validate_sections"),
        ("validate_sections", "render"),
        ("render", "validate_report"),
        ("fallback", "validate_report"),
        ("validate_report", END),
    ):
        graph.add_conditional_edges(
            source, _route_error, {"next": target, "error": "failure"}
        )
    graph.add_edge("failure", END)
    return graph.compile()


def report_writer_agent(state: GlobalState) -> dict[str, Any]:
    """GlobalState를 내부 그래프에 전달하고 final_report 문자열만 반환."""
    try:
        result = build_report_graph().invoke({"request": deepcopy(state)})
        return result["output"]
    except Exception as error:  # noqa: BLE001 - 그래프 실행 경계의 오류 처리
        return _report_failure({"error_type": type(error).__name__})["output"]
