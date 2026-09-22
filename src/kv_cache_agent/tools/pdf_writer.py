"""Markdown 평가 보고서를 한국어 지원 PDF로 변환하는 도구."""

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
)

FONT_CANDIDATES = (
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
    Path("/Library/Fonts/Arial Unicode.ttf"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
)
FONT_NAME = "KVCacheKorean"


def _find_korean_font() -> Path:
    """실행 환경에서 사용할 수 있는 한국어 글꼴 경로를 찾는다."""
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    candidates = ", ".join(str(path) for path in FONT_CANDIDATES)
    raise RuntimeError(
        "한국어 PDF 글꼴을 찾을 수 없습니다. 다음 경로 중 하나에 글꼴을 설치하세요: "
        + candidates
    )


def _register_korean_font() -> str:
    """ReportLab에 한국어 글꼴을 등록하고 글꼴 이름을 반환한다."""
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(_find_korean_font())))
    return FONT_NAME


def _clean_inline_markdown(text: str) -> str:
    """단순 Markdown 표기를 PDF 본문용 일반 텍스트로 정리한다."""
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`([^`]*)`", r"\1", text)
    return escape(text)


def _styles(font_name: str) -> dict[str, ParagraphStyle]:
    """보고서 계층별 ParagraphStyle을 구성한다."""
    return {
        "h1": ParagraphStyle(
            "ReportH1",
            fontName=font_name,
            fontSize=17,
            leading=23,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=12,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2",
            fontName=font_name,
            fontSize=12,
            leading=17,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#245B7A"),
            spaceBefore=8,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "ReportBody",
            fontName=font_name,
            fontSize=9.5,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#202124"),
            spaceAfter=6,
        ),
        "interpretation": ParagraphStyle(
            "ReportInterpretation",
            parent=ParagraphStyle(
                "ReportInterpretationBase",
                fontName=font_name,
                fontSize=9.5,
                leading=15,
                alignment=TA_LEFT,
                textColor=colors.HexColor("#1F4E79"),
                leftIndent=5,
                spaceAfter=6,
            ),
        ),
        "limitation": ParagraphStyle(
            "ReportLimitation",
            fontName=font_name,
            fontSize=9.5,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#7A4E00"),
            leftIndent=5,
            spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet",
            fontName=font_name,
            fontSize=9.5,
            leading=15,
            alignment=TA_LEFT,
            leftIndent=10,
            firstLineIndent=-7,
            textColor=colors.HexColor("#202124"),
            spaceAfter=4,
        ),
        "footer": ParagraphStyle(
            "ReportFooter",
            fontName=font_name,
            fontSize=8,
            leading=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#6B7280"),
        ),
    }


def _paragraph_style(line: str, styles: dict[str, ParagraphStyle]) -> ParagraphStyle:
    """본문 접두어에 따라 일반·해석·한계 스타일을 선택한다."""
    if line.startswith("해석:"):
        return styles["interpretation"]
    if line.startswith("한계:"):
        return styles["limitation"]
    if line.startswith("- "):
        return styles["bullet"]
    return styles["body"]


def _markdown_to_flowables(markdown_text: str, styles: dict[str, ParagraphStyle]):
    """현재 보고서 Markdown의 제목·본문·목록을 ReportLab 요소로 변환한다."""
    flowables = []
    paragraph_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text = " ".join(line.strip() for line in paragraph_lines)
        style = _paragraph_style(text, styles)
        if text.startswith("- "):
            text = "- " + text[2:].lstrip()
        flowables.append(Paragraph(_clean_inline_markdown(text), style))
        paragraph_lines.clear()

    for raw_line in markdown_text.splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        if line.startswith("# "):
            flush_paragraph()
            flowables.append(Paragraph(_clean_inline_markdown(line[2:]), styles["h1"]))
            flowables.append(
                HRFlowable(
                    width="100%",
                    thickness=0.5,
                    color=colors.HexColor("#D6E2EA"),
                    spaceAfter=7,
                )
            )
            continue
        if line.startswith("## "):
            flush_paragraph()
            flowables.append(Paragraph(_clean_inline_markdown(line[3:]), styles["h2"]))
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return flowables


def _draw_footer(font_name: str):
    """페이지 하단에 문서명과 페이지 번호를 그리는 콜백을 반환한다."""

    def draw(canvas, document) -> None:
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(18 * mm, 10 * mm, "KV Cache SW vs HW 비교 평가")
        canvas.drawRightString(
            A4[0] - 18 * mm,
            10 * mm,
            f"{document.page}",
        )
        canvas.restoreState()

    return draw


def write_pdf(markdown_text: str, output_path: str | Path) -> Path:
    """Markdown 보고서를 한국어 PDF로 저장하고 생성 경로를 반환한다."""
    if not markdown_text.strip():
        raise ValueError("PDF로 변환할 보고서 내용이 없습니다.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    font_name = _register_korean_font()
    styles = _styles(font_name)
    document = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=18 * mm,
        title="KV Cache SW vs HW 비교 평가",
        author="SKALA LangGraph",
    )
    story = _markdown_to_flowables(markdown_text, styles)
    if not story:
        raise ValueError("PDF로 변환할 본문 요소가 없습니다.")
    document.build(story, onFirstPage=_draw_footer(font_name), onLaterPages=_draw_footer(font_name))
    return output
