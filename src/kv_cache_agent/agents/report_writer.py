"""보고서 생성 에이전트: 종합 결과의 문서화와 출처 연결.

인풋:
    GlobalState의 user_query, synthesis_result, verification_result,
    evidence_cards, 선택 입력 research_plan.
    목차와 절별 지침은 prompts/report_writer.yaml에서 로드.

함수 기능:
    report_writer_agent: 입력 확인 → LLM 본문 작성 → 인용 검사 → 문서 조립.
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
        evidence_cards: list[EvidenceCard]
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
from pathlib import Path
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from kv_cache_agent.agents.synthesis import (
    _load_config,
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


def report_writer_agent(state: GlobalState) -> dict[str, Any]:
    """종합 결과만 문서화. 관점별 원본 요약의 재평가와 신규 검색 없음."""
    try:
        config = _load_prompt_config()
        result = state.get("synthesis_result", {})
        if result.get("status") == "failed":
            return {"final_report": "# 보고서 생성 실패\n평가 종합 실패로 작성 중단\n"}
        if result.get("status") not in ("ok", "insufficient_evidence"):
            return {"final_report": _fallback(config, "평가 종합 결과 미확보", [])}
        allowed, limitations = _verified_cards(state)
        ids = result.get("evidence_ids", [])
        if not isinstance(ids, list) or not set(ids) <= allowed.keys():
            raise ValueError("Invalid synthesis evidence IDs")
        cards = {key: allowed[key] for key in ids}
        if not cards:
            return {
                "final_report": _fallback(
                    config,
                    "검증 완료 근거 부족으로 판단 보류",
                    [*limitations, *result.get("limitations", [])],
                )
            }
        outline = _body_sections(config)
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
        markdown = _render_markdown(config, sections, cards)
        if re.findall(r"^# (.+)$", markdown, re.MULTILINE) != TITLES:
            raise ValueError("Rendered outline mismatch")
        return {"final_report": markdown}
    except Exception as error:  # noqa: BLE001 - 노드 경계에서 실패 결과로 변환
        return {
            "final_report": f"# 보고서 생성 실패\n보고서 처리 오류 ({type(error).__name__})\n"
        }
