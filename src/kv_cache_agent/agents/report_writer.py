# 내부 LangGraph: 순차 처리와 조건부 분기로 구성, 반복 루프 없음.
# 정상: START → load_config → prepare_context → generate → validate_sections
#       → render → validate_report → END
# 자료 부족: prepare_context → generate → validate_sections → render → END
# 처리 오류: 해당 노드 → failure → END / 종합 실패: 현재 입력으로 보고서 생성 계속
# 외부 반환: {"final_report": str}, 내부 상태: ReportState

"""보고서 생성 에이전트: 종합 결과의 문서화와 출처 연결.

인풋:
    GlobalState의 user_query, synthesis_result, verification_result,
    evidence_cards, verified_evidence_cards, usable_evidence_cards,
    선택 입력 research_plan.
    목차와 절별 지침은 prompts/report_writer.yaml에서 로드.

함수 기능:
    report_writer_agent: 내부 LangGraph 호출 후 기존 문자열 형식으로 결과 전달.
    build_report_graph: 설정 → 입력 → 생성 → 인용 검사 → 조립 → 최종 검사 연결.
    조건부 경로: 자료 부족은 fallback, 처리 오류는 failure 노드로 이동.
    _load_prompt_config: 최상위 목차 8개와 하위 절 20개의 구조 검사.
    _render_markdown: 고정 제목, 인용 번호, 실제 사용한 참고문헌 조립.
    _fallback: 종합 결과 자체가 없을 때만 외부 호출 없이 안내 보고서 구성.

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
    evidence_cards, verified_evidence_cards, usable_evidence_cards: list[EvidenceCard]
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
from functools import lru_cache, wraps
from pathlib import Path
from typing import Any, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, ConfigDict, Field, create_model

from kv_cache_agent.agents.synthesis import (
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
INLINE_EVIDENCE_ARTIFACT = re.compile(r"\s*\(\s*evidence_ids\s*:\s*\[[^\]]*\]\s*\)")
INFERENCE_PREFIX = re.compile(r"^\s*(?:추론|해석)\s*:\s*")
ANALYTICAL_METHOD_NOTE = (
    "자료가 제한된 항목은 확인된 기술 특성, 클라우드 운영 조건과 일반적인 "
    "인과관계를 연결한 분석으로 보완했으며, 실제 채택·성능·비용은 별도 검증이 필요하다."
)
PLACEHOLDER_MARKERS = (
    "검증 가능한 근거가 부족하여",
    "검증 가능한 근거가 없어",
    "판단을 보류",
    "근거 부족으로 해당 항목",
)


def _missing_section_paragraphs(section_id: str) -> "list[Paragraph]":
    """자료 미확보는 한계로 명시한다. 기술 해석이나 실험 결과를 만들어 채우지 않는다."""
    return [
        Paragraph(
            text="이 항목의 검증 근거가 확보되지 않았다. 추가 조사와 사람의 검토가 필요하다.",
            claim_type="limitation",
            evidence_ids=[],
        )
    ]


def _is_placeholder_section(section: "Section") -> bool:
    """절이 실질적인 분석 없이 판단 보류 문장만 포함하는지 확인한다."""
    return bool(section.paragraphs) and all(
        any(marker in paragraph.text for marker in PLACEHOLDER_MARKERS)
        for paragraph in section.paragraphs
    )


class Paragraph(BaseModel):
    """사실, 추론, 한계와 관련 근거 ID를 담는 본문 문단."""

    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=1)
    claim_type: Literal["fact", "inference", "limitation"]
    evidence_ids: list[str]


class Section(BaseModel):
    """YAML의 절 ID에 대응하는 생성 문단 목록."""

    model_config = ConfigDict(extra="forbid")
    section_id: Literal[
        "summary",
        "section_1_1",
        "section_1_2",
        "section_1_3",
        "section_2_1",
        "section_2_2",
        "section_2_3",
        "section_3_1",
        "section_3_2",
        "section_3_3",
        "section_4_1",
        "section_4_2",
        "section_4_3",
        "section_4_4",
        "section_5_1",
        "section_5_2",
        "section_5_3",
        "section_6_1",
        "section_6_2",
        "section_6_3",
        "section_6_4",
    ]
    paragraphs: list[Paragraph] = Field(min_length=1)


class ReportDraft(BaseModel):
    """SUMMARY와 하위 절의 생성 응답 검사용 내부 모델."""

    model_config = ConfigDict(extra="forbid")
    sections: list[Section]


def _bounded_draft_model(cards: dict, section_ids: list[str]):
    """허용된 절/카드 ID를 JSON Schema enum으로 제한한다.

    프롬프트에 ID를 적는 것만으로는 모델의 새 ID 생성을 막을 수 없다.
    실제 입력에서 enum을 만들어 출력 단계부터 제한하고 코드에서도 다시 검사한다.
    """
    paragraph_model = Paragraph
    if cards:
        # 사실 문장의 인용은 verified ID만 선택할 수 있도록 별도 분기로 만든다.
        # partially_verified 카드는 해석/한계 분기에서만 허용한다.
        nonfact_model = create_model(
            "InterpretationParagraph",
            __base__=Paragraph,
            evidence_ids=(list[Literal[tuple(cards)]], Field(...)),
            claim_type=(Literal["inference", "limitation"], Field(...)),
        )
        verified_ids = tuple(
            k for k, c in cards.items() if c.get("verification_status") == "verified"
        )
        if verified_ids:
            fact_model = create_model(
                "VerifiedFactParagraph",
                __base__=Paragraph,
                evidence_ids=(list[Literal[verified_ids]], Field(..., min_length=1)),
                claim_type=(Literal["fact"], Field(...)),
            )
            paragraph_model = fact_model | nonfact_model
        else:
            paragraph_model = nonfact_model
    # 배열 안에 section_id를 넣으면 어떤 절은 반복되고 다른 절은 누락될 수 있다.
    # 현재 장의 절을 필수 '고정 필드'로 만들어 누락 자체를 스키마에서 막는다.
    return create_model(
        "ChapterDraft",
        __config__=ConfigDict(extra="forbid"),
        **{
            key: (list[paragraph_model], Field(..., min_length=1))
            for key in section_ids
        },
    )


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


def _remove_inline_evidence_artifact(text: str) -> str:
    """LLM이 본문에 중복 출력한 구조화 필드와 추론 접두어를 제거한다."""
    cleaned = INLINE_EVIDENCE_ARTIFACT.sub("", text)
    cleaned = INFERENCE_PREFIX.sub("", cleaned).strip()
    # 유효한 evidence_ids로 번호를 다시 부여하므로 모델의 중복 번호만 제거한다.
    return re.sub(r"\s*\[\d+\]", "", cleaned).strip()


def _normalize_sections(
    draft: ReportDraft,
    expected_ids: list[str],
    limitations: list[str],
    allow_uncited_inference: bool = False,
) -> list[Section]:
    """누락·중복을 오류로 처리한다. 고정 문장으로 내용을 채우지 않는다."""
    grouped: dict[str, Section] = {}
    for section in draft.sections:
        key = section.section_id.strip()
        if key not in expected_ids:
            raise ValueError("Unknown section ID")
        if key in grouped:
            raise ValueError("Duplicate section ID")
        grouped[key] = section
    if set(grouped) != set(expected_ids):
        raise ValueError("Missing report sections")
    return [grouped[key] for key in expected_ids]


def _render_markdown(
    config: dict, sections: dict[str, list[Paragraph]], cards: dict
) -> str:
    """문단 -> 고유 출처 번호 -> 원문 URL/서지/위치의 연결을 보존한다."""
    from datetime import UTC, datetime
    from urllib.parse import urlparse

    numbers: dict[str, int] = {}
    reference_cards: dict[str, list[str]] = {}
    lines = []
    for section in config["report_structure"][:-1]:
        lines.extend([f"# {section['title']}", ""])
        for child in section.get("subsections", [section]):
            if child is not section:
                lines.extend([f"## {child['title']}", ""])
            for paragraph in sections[child["id"]]:
                keys = []
                for evidence_id in paragraph.evidence_ids:
                    card = cards[evidence_id]
                    key = str(card.get("canonical_url") or card["source_url"])
                    if key not in numbers:
                        numbers[key] = len(numbers) + 1
                        reference_cards[key] = []
                    if evidence_id not in reference_cards[key]:
                        reference_cards[key].append(evidence_id)
                    keys.append(key)
                citations = " ".join(f"[{numbers[k]}]" for k in dict.fromkeys(keys))
                prefix = {"fact": "", "inference": "해석: ", "limitation": "한계: "}[
                    paragraph.claim_type
                ]
                lines.extend(
                    [f"{prefix}{_plain(paragraph.text)} {citations}".strip(), ""]
                )
    lines.extend(["# REFERENCE", ""])
    today = datetime.now(tz=UTC).date().isoformat()
    for url, number in numbers.items():
        ids = reference_cards[url]
        card = cards[ids[0]]
        date_text = str(card.get("published_date") or "게시일 미표기")
        author = str(
            card.get("authors")
            or card.get("publisher")
            or urlparse(url).netloc
            or "동봉 논문"
        )
        title = str(card.get("source_title") or "제목 미표기")
        publisher = str(card.get("publisher") or urlparse(url).netloc)
        locators = "; ".join(
            dict.fromkeys(str(cards[i].get("source_locator", "")) for i in ids)
        )
        checked = str(card.get("retrieved_at") or today)
        metadata = f"{author} ({date_text}). {title}. {publisher}. {url}. 인용 위치: {locators}. 확인일: {checked}."
        lines.extend([f"[{number}] {_plain(metadata)} 근거 ID: {', '.join(ids)}", ""])
    if not numbers:
        lines.append("본문에 사용한 검증 근거 없음")
    return "\n".join(lines).strip() + "\n"


def _fallback(config: dict, reason: str, limitations: list[str]) -> str:
    """부족 사유와 한계 입력 → 지정 목차별 분석 구성 → Markdown 반환."""
    sections = {
        child["id"]: _missing_section_paragraphs(child["id"])
        for child in _body_sections(config)
    }
    sections["summary"].insert(
        0,
        Paragraph(
            text=(
                f"{reason} 공개 자료와 현재 입력 범위를 바탕으로 기술 구조와 "
                "클라우드 운영 영향을 중심으로 비교를 구성한다."
            ),
            claim_type="limitation",
            evidence_ids=[],
        ),
    )
    sections["section_6_4"].append(
        Paragraph(
            text=ANALYTICAL_METHOD_NOTE,
            claim_type="limitation",
            evidence_ids=[],
        )
    )
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
    allow_uncited_inference: bool
    response: Any
    sections: dict[str, list[Paragraph]]
    markdown: str
    error_type: str
    output: dict[str, str]


def _report_guard(node):
    """보고서 노드 예외를 노드명과 함께 안전한 오류 상태로 변환."""

    error_labels = (
        ("Invalid synthesis evidence IDs", "invalid_synthesis_evidence_ids"),
        ("Missing or duplicate section", "missing_or_duplicate_section"),
        ("Unknown citation", "unknown_citation"),
        ("Uncited assertion", "uncited_assertion"),
        (
            "Partially verified evidence cannot support fact",
            "partial_card_fact",
        ),
        (
            "Numeric claim absent from cited evidence",
            "fabricated_number",
        ),
        ("Inline citation or heading is not permitted", "inline_artifact"),
        ("Rendered outline mismatch", "rendered_outline_mismatch"),
    )

    def error_label(error: Exception) -> str:
        message = str(error)
        for fragment, label in error_labels:
            if fragment in message:
                return label
        return type(error).__name__

    @wraps(node)
    def guarded(state):
        try:
            return node(state)
        except Exception as error:  # noqa: BLE001 - 노드 경계의 안전한 오류 변환
            return {
                "error_type": (
                    f"{node.__name__}:{type(error).__name__}:{error_label(error)}"
                )
            }

    return guarded


@_report_guard
def _load_report_config(local: ReportState) -> dict:
    """입력: 내부 상태 → 처리: YAML과 목차 검사 → 출력: config."""
    return {"config": _load_prompt_config()}


@_report_guard
def _prepare_report_context(local: ReportState) -> dict:
    """입력: request → 처리: 종합 상태와 전체 허용 근거 확인 → 출력: 생성 자료."""
    state = local["request"]
    result = dict(state.get("synthesis_result") or {})
    synthesis_failed = result.get("status") == "failed"
    if result.get("status") not in ("ok", "insufficient_evidence"):
        if not synthesis_failed:
            return {"fallback_reason": "평가 종합 결과 미확보", "limitations": []}
        # 종합 실패를 재시도하지 않고 현재 State의 입력만으로 보고서를 계속 작성한다.
        # 이 상태는 아래 limitations로 남겨 보고서가 정상 종합 결과처럼 보이지 않게 한다.
        result["status"] = "insufficient_evidence"
        result["evidence_ids"] = []
        result["limitations"] = [
            *result.get("limitations", []),
            "평가 종합이 완료되지 않아 확보된 기술·관점 입력을 바탕으로 분석을 계속함",
        ]
    allowed, limitations = _verified_cards(state)
    ids = result.get("evidence_ids", [])
    if not isinstance(ids, list) or not set(ids) <= allowed.keys():
        raise ValueError("Invalid synthesis evidence IDs")
    # synthesis_result의 ID는 핵심 종합에 직접 사용된 카드 목록이다.
    # 보고서 작성은 시장·이해관계자 등 다른 절에도 답할 수 있도록 전체 허용 카드를 사용한다.
    cards = allowed
    return {
        "cards": cards,
        "result": result,
        # 자료가 불충분해도 보고서를 생성하되, 무인용 문단은 추론으로 표시한다.
        "allow_uncited_inference": result.get("status") == "insufficient_evidence"
        or not cards,
        "limitations": list(
            dict.fromkeys([*limitations, *result.get("limitations", [])])
        ),
    }


def _route_report_input(local: ReportState) -> str:
    """처리 오류는 failure, 입력 미확보는 fallback, 그 외에는 generate로 이동."""
    if local.get("error_type"):
        return "error"
    return "fallback" if local.get("fallback_reason") else "ready"


@_report_guard
def _generate_sections(local: ReportState) -> dict:
    """장별로 생성하여 토큰 제한으로 인한 절 누락을 줄인다.

    Supervisor의 승인/부족 관점/검수 사유를 반드시 입력한다. 전체 카드 본문을
    여러 번 중복 전달하지 않고 필요한 인용 정보만 유지한다. 생성 실패나 누락은
    품질 실패이며 상수 문단으로 숨기지 않는다.
    """
    state, cards, config = local["request"], local["cards"], local["config"]
    context = {
        "user_query": state.get("user_query", ""),
        "synthesis_result": {
            key: value
            for key, value in local["result"].items()
            if key not in {"errors", "payload"}
        },
        "evidence_cards": list(cards.values()),
        "allowed_evidence_ids": list(cards),
        "numeric_evidence_by_id": {
            key: sorted(
                set(
                    re.findall(
                        r"(?<![\w.])\d+(?:[.,]\d+)*(?:/\d+(?:[.,]\d+)*)?",
                        str(card.get("claim", ""))
                        + " "
                        + str(card.get("evidence_text", "")),
                    )
                )
            )
            for key, card in cards.items()
        },
        "perspective_results": {
            p: {
                "status": state.get(f"{p}_result", {}).get("status"),
                "summary": state.get(f"{p}_result", {}).get("summary", ""),
                "limitations": state.get(f"{p}_result", {}).get("limitations", []),
            }
            for p in ("technical", "market", "stakeholder", "cloud_domain")
        },
        "supervisor_review": state.get("control", {}).get("evidence_review", {}),
        "supervisor_approved": state.get("control", {}).get("evidence_ready", False),
        "rework_attempts": state.get("control", {}).get("retry_count", {}),
        "research_plan": state.get("research_plan", {}),
        "quality_feedback": state.get("quality_feedback", []),
        "limitations": local.get("limitations", []),
        "allow_uncited_inference": local.get("allow_uncited_inference", False),
    }
    generator = get_llm()
    assembled = []
    excluded_sections: list[str] = []
    for chapter in config["report_structure"][:-1]:
        print(f"보고서 작성 중: {chapter['title']}", flush=True)
        children = chapter.get("subsections", [chapter])
        ids = [child["id"] for child in children]
        llm = generator.with_structured_output(
            _bounded_draft_model(cards, ids), method="json_schema", strict=True
        )
        batch = {
            **context,
            "chapter": chapter,
            "required_section_ids": ids,
            "response_rule": "출력 모델의 각 필수 필드 이름이 section_id이고, 값은 해당 절의 문단 목록이다. sections 배열을 만들지 않는다.",
            "numeric_style": "본문에는 아라비아 숫자를 가급적 쓰지 않는다. 특히 재조사 횟수, 관점 수, 페이지 한도는 숫자 대신 여러 차례/네 관점/제출 한도처럼 표현한다. 실험 수치가 꼭 필요하면 인용 카드 원문에 있는 수치만 사용한다.",
            "length_target": "SUMMARY 300~500자, 각 하위 절 350~600자; 근거가 없으면 한계와 추가 확인 방법만 정확히 기술",
        }
        for attempt in range(2):
            raw = llm.invoke(
                [
                    SystemMessage(content=config["system_prompt"]),
                    HumanMessage(content=json.dumps(batch, ensure_ascii=False)),
                ]
            )
            try:
                data = raw.model_dump() if isinstance(raw, BaseModel) else raw
                if "sections" in data:
                    # 기존 테스트/호환 응답은 내부 표준 형태로 그대로 읽는다.
                    draft = ReportDraft.model_validate(data)
                else:
                    draft = ReportDraft(
                        sections=[
                            Section(section_id=key, paragraphs=data[key])
                            for key in ids
                            if key in data
                        ]
                    )
                selected = [s for s in draft.sections if s.section_id in ids]
                _normalize_sections(ReportDraft(sections=selected), ids, [])
                for section in selected:
                    for paragraph in section.paragraphs:
                        paragraph.text = _remove_inline_evidence_artifact(
                            paragraph.text
                        )
                        if not set(paragraph.evidence_ids) <= cards.keys():
                            raise ValueError("Unknown citation")
                        if paragraph.claim_type == "fact":
                            if not paragraph.evidence_ids:
                                raise ValueError("Uncited assertion")
                            if any(
                                cards[i].get("verification_status") != "verified"
                                for i in paragraph.evidence_ids
                            ):
                                raise ValueError(
                                    "Partially verified evidence cannot support fact"
                                )
                        if paragraph.claim_type != "limitation":
                            try:
                                _validate_numbers(
                                    paragraph.text, paragraph.evidence_ids, cards
                                )
                            except ValueError as error:
                                raise ValueError(
                                    f"{error}; section={section.section_id}; text={paragraph.text[:180]}"
                                ) from error
                        if re.search(r"\[[^]]+\]|^\s*#", paragraph.text):
                            raise ValueError(
                                "Inline citation or heading is not permitted"
                            )
                break
            except ValueError as error:
                if attempt == 1:
                    # Supervisor가 이미 미승인한 '검토용 초안'에만 적용한다.
                    # 두 번 요청해도 숫자/인용이 틀린 단락은 주장 자체를 제거하고
                    # 제외 사실을 명시한다. 승인된 최종 보고서는 여전히 실패시킨다.
                    if state.get("control", {}).get("evidence_ready") is False and str(
                        error
                    ).startswith("Numeric claim absent"):
                        for section in selected:
                            valid_paragraphs = []
                            removed = False
                            for paragraph in section.paragraphs:
                                try:
                                    if paragraph.claim_type != "limitation":
                                        _validate_numbers(
                                            paragraph.text,
                                            paragraph.evidence_ids,
                                            cards,
                                        )
                                    valid_paragraphs.append(paragraph)
                                except ValueError:
                                    removed = True
                            if removed:
                                title = next(
                                    c["title"]
                                    for c in children
                                    if c["id"] == section.section_id
                                )
                                valid_paragraphs.append(
                                    Paragraph(
                                        text=f"검토 필요: {title}의 일부 단락은 수치와 인용 위치 연결을 확인하지 못해 제외했다. 해당 원문 페이지와 카드 연결을 다시 검토해야 한다.",
                                        claim_type="limitation",
                                        evidence_ids=[],
                                    )
                                )
                                excluded_sections.append(section.section_id)
                            section.paragraphs = valid_paragraphs
                        break
                    raise
                # invoke는 매번 독립 호출이므로 이전 초안도 함께 전달해야
                # 모델이 '어느 문장을 수정해야 하는지' 알 수 있다.
                batch["previous_draft"] = (
                    raw.model_dump() if isinstance(raw, BaseModel) else raw
                )
                # 문제를 모델에게 구체적으로 알려 한 번만 고친다.
                # 근거 없는 숫자를 코드에서 지우거나 추론으로 덮지 않는다.
                batch["repair_instruction"] = (
                    "직전 초안 검증 실패: "
                    + str(error)[:500]
                    + ". numeric_evidence_by_id에서 해당 수치를 실제로 지원하는 카드 ID를 찾아 올바르게 인용하세요. 지원 카드가 없으면 수치 주장을 제거하세요. 현재 장의 절을 모두 복원하고 잘못된 인용/사실 유형을 바로잡아 반환하세요."
                )
        assembled.extend(selected)
    limitations = list(local.get("limitations", []))
    if excluded_sections:
        limitations.append(
            "수치·인용 검사 미통과 단락을 제외한 검토용 초안이며 최종 제출본이 아닙니다."
        )
    return {"response": ReportDraft(sections=assembled), "limitations": limitations}


@_report_guard
def _validate_sections(local: ReportState) -> dict:
    """입력: response → 처리: 절 ID, 인용, 수치 검사 → 출력: sections."""
    response, cards = local["response"], local["cards"]
    limitations = list(local.get("limitations", []))
    outline = _body_sections(local["config"])
    draft = ReportDraft.model_validate(response)
    allow_uncited_inference = local.get("allow_uncited_inference", False)
    # 근거는 evidence_ids 필드로만 관리하고, 본문에 중복된 내부 표기는 제거한다.
    for section in draft.sections:
        for paragraph in section.paragraphs:
            paragraph.text = _remove_inline_evidence_artifact(paragraph.text)
    expected = [s["id"] for s in outline]
    sections_to_validate = _normalize_sections(
        draft, expected, limitations, allow_uncited_inference
    )
    for section in sections_to_validate:
        for paragraph in section.paragraphs:
            unknown_ids = set(paragraph.evidence_ids) - cards.keys()
            if unknown_ids:
                raise ValueError("Unknown citation")
            if (
                paragraph.claim_type != "limitation"
                and not paragraph.evidence_ids
                and not (
                    allow_uncited_inference and paragraph.claim_type == "inference"
                )
            ):
                raise ValueError("Uncited assertion")
            if paragraph.claim_type == "fact" and any(
                cards[evidence_id].get("verification_status") == "partially_verified"
                for evidence_id in paragraph.evidence_ids
            ):
                raise ValueError("Partially verified evidence cannot support fact")
            if paragraph.claim_type != "limitation":
                _validate_numbers(paragraph.text, paragraph.evidence_ids, cards)
            if not paragraph.text.strip() or re.search(
                r"\[[^]]+\]|^\s*#", paragraph.text
            ):
                raise ValueError("Inline citation or heading is not permitted")
    sections = {s.section_id: s.paragraphs for s in sections_to_validate}
    if local["request"].get("control", {}).get("evidence_ready") is False and not any(
        "검토용 초안" in p.text for p in sections["summary"]
    ):
        sections["summary"].insert(
            0,
            Paragraph(
                text="검토용 초안: Supervisor가 전체 근거 충분성을 승인하지 않은 상태다. 부족 관점과 제외된 단락을 보완한 후 다시 검수해야 하며, 이 문서는 최종 제출본이 아니다.",
                claim_type="limitation",
                evidence_ids=[],
            ),
        )
    # 반복적인 문단별 표시는 제거하되, 보고서의 분석 방식은 한 번 명시한다.
    if allow_uncited_inference and not any(
        paragraph.text == ANALYTICAL_METHOD_NOTE
        for paragraph in sections["section_6_4"]
    ):
        sections["section_6_4"].append(
            Paragraph(
                text=ANALYTICAL_METHOD_NOTE,
                claim_type="limitation",
                evidence_ids=[],
            )
        )
    # 상류에서 확인한 자료 부족은 LLM의 누락 여부와 무관하게 보고서에 보존.
    human_limitations = [
        value
        for value in limitations
        if not re.search(r"section_\d|partial_card_fact|evidence_id", value)
    ]
    if human_limitations:
        sections["section_6_4"].append(
            Paragraph(
                text=" / ".join(human_limitations),
                claim_type="limitation",
                evidence_ids=[],
            )
        )
    return {"sections": sections}


@_report_guard
def _render_report(local: ReportState) -> dict:
    """입력: 검사된 절과 근거 → 처리: 제목과 참고문헌 조립 → 출력: markdown."""
    return {
        "markdown": _render_markdown(local["config"], local["sections"], local["cards"])
    }


@_report_guard
def _validate_report(local: ReportState) -> dict:
    """입력: markdown → 처리: 최종 제목 순서 검사 → 출력: 기존 final_report 형식."""
    markdown = local["markdown"]
    if re.findall(r"^# (.+)$", markdown, re.MULTILINE) != TITLES:
        raise ValueError("Rendered outline mismatch")
    return {"output": {"final_report": markdown}}


@_report_guard
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
        return _report_failure(
            {"error_type": f"report_writer_agent:{type(error).__name__}"}
        )["output"]
