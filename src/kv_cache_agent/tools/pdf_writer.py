"""보고서를 동봉한 한글 폰트와 장별 레이아웃으로 출력한다.

본문을 새로 생성하거나 페이지 수를 채우기 위한 내용을 추가하지 않는다.
전체 정식 목차에서는 장마다 새 페이지를 시작하고 짧은 문서는 연속 배치한다.
"""

import re
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import HRFlowable, PageBreak, Paragraph, SimpleDocTemplate

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "fonts"
FONT_NAME = "KVCacheKorean"
BOLD_NAME = "KVCacheKoreanBold"
FONT_CANDIDATES = (
    ASSETS / "NanumGothic-Regular.ttf",
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/System/Library/Fonts/Supplemental/AppleGothic.ttf"),
)


def _find_korean_font() -> Path:
    """OS별 차이를 줄이기 위해 동봉한 OFL 한글 폰트를 우선 사용한다."""
    for path in FONT_CANDIDATES:
        if path.is_file():
            return path
    raise RuntimeError("한국어 폰트가 없습니다. assets/fonts의 동봉 파일을 확인하세요.")


def _register_korean_font() -> str:
    if FONT_NAME not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(TTFont(FONT_NAME, str(_find_korean_font())))
        bold = ASSETS / "NanumGothic-Bold.ttf"
        pdfmetrics.registerFont(
            TTFont(BOLD_NAME, str(bold if bold.is_file() else _find_korean_font()))
        )
        pdfmetrics.registerFontFamily(FONT_NAME, normal=FONT_NAME, bold=BOLD_NAME)
    return FONT_NAME


def _clean_inline_markdown(text: str) -> str:
    text = re.sub(r"`([^`]*)`", r"\1", text)
    text = escape(text)
    return re.sub(r"\*\*(.*?)\*\*", r"<b>\1</b>", text)


def _styles(font_name: str) -> dict[str, ParagraphStyle]:
    base = ParagraphStyle(
        "Body",
        fontName=font_name,
        fontSize=10.2,
        leading=17,
        wordWrap="CJK",
        textColor=colors.HexColor("#253348"),
        spaceAfter=9,
        splitLongWords=True,
    )
    return {
        "body": base,
        "h1": ParagraphStyle(
            "Chapter",
            parent=base,
            fontName=BOLD_NAME,
            fontSize=18,
            leading=25,
            spaceBefore=0,
            spaceAfter=12,
            textColor=colors.HexColor("#142F49"),
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Section",
            parent=base,
            fontName=BOLD_NAME,
            fontSize=11.2,
            leading=17,
            spaceBefore=12,
            spaceAfter=7,
            textColor=colors.HexColor("#126D7C"),
            keepWithNext=True,
        ),
        "interpretation": ParagraphStyle("Interpretation", parent=base),
        "limitation": ParagraphStyle(
            "Limitation",
            parent=base,
            leftIndent=8,
            borderColor=colors.HexColor("#D8B67A"),
            borderWidth=0.4,
            borderPadding=7,
            backColor=colors.HexColor("#FBF7ED"),
            spaceAfter=12,
        ),
        "reference": ParagraphStyle(
            "Reference", parent=base, fontSize=8.3, leading=13, spaceAfter=10
        ),
        "title": ParagraphStyle(
            "Title",
            parent=base,
            fontName=BOLD_NAME,
            fontSize=23,
            leading=32,
            spaceAfter=9,
            textColor=colors.HexColor("#142F49"),
        ),
        "label": ParagraphStyle(
            "Label",
            parent=base,
            fontSize=9,
            leading=14,
            textColor=colors.HexColor("#64748B"),
            spaceAfter=12,
        ),
    }


def _paragraph_style(line: str, styles: dict[str, ParagraphStyle]) -> ParagraphStyle:
    if line.startswith("한계:"):
        return styles["limitation"]
    if line.startswith("해석:"):
        return styles["interpretation"]
    return styles["body"]


def _markdown_to_flowables(markdown_text: str, styles: dict[str, ParagraphStyle]):
    """제목을 다음 본문과 함께 배치하여 페이지 끝의 고립된 제목을 막는다."""
    flowables = []
    paragraph_lines: list[str] = []
    complete_outline = len(re.findall(r"(?m)^# ", markdown_text)) >= 8
    in_references = False
    chapter_count = 0

    def flush():
        if paragraph_lines:
            text = " ".join(paragraph_lines)
            style = (
                styles["reference"] if in_references else _paragraph_style(text, styles)
            )
            flowables.append(Paragraph(_clean_inline_markdown(text), style))
            paragraph_lines.clear()

    if complete_outline:
        flowables.extend(
            [
                Paragraph("KV CACHE / TECHNOLOGY ASSESSMENT", styles["label"]),
                Paragraph("KV Cache 최적화 기술 평가", styles["title"]),
                Paragraph(
                    "TurboQuant · CXL-Hybrid ITME<br/>Supervisor 기반 다관점 분석",
                    styles["label"],
                ),
                HRFlowable(
                    width="100%",
                    thickness=2,
                    color=colors.HexColor("#126D7C"),
                    spaceAfter=17,
                ),
            ]
        )
    for raw in markdown_text.splitlines():
        line = raw.strip()
        if not line:
            flush()
        elif line.startswith("# "):
            flush()
            if complete_outline and chapter_count:
                flowables.append(PageBreak())
            chapter_count += 1
            in_references = line[2:] == "REFERENCE"
            flowables.append(Paragraph(_clean_inline_markdown(line[2:]), styles["h1"]))
            if complete_outline and chapter_count == 1:
                flowables.append(
                    Paragraph(
                        "출처에 직접 확인된 설명과 '해석:', '한계:'로 표시한 분석을 구분해 읽으세요.",
                        styles["label"],
                    )
                )
        elif line.startswith("## "):
            flush()
            flowables.append(Paragraph(_clean_inline_markdown(line[3:]), styles["h2"]))
        else:
            paragraph_lines.append(line)
    flush()
    return flowables


def _draw_footer(font_name: str):
    def draw(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#DDE5ED"))
        canvas.line(20 * mm, 17 * mm, A4[0] - 20 * mm, 17 * mm)
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#64748B"))
        canvas.drawString(20 * mm, 11 * mm, "SKALA  /  KV Cache 다관점 평가")
        canvas.drawRightString(A4[0] - 20 * mm, 11 * mm, str(document.page))
        canvas.restoreState()

    return draw


def write_pdf(markdown_text: str, output_path: str | Path) -> Path:
    if not markdown_text.strip():
        raise ValueError("PDF로 변환할 보고서 내용이 없습니다.")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    font = _register_korean_font()
    doc = SimpleDocTemplate(
        str(output),
        pagesize=A4,
        rightMargin=20 * mm,
        leftMargin=20 * mm,
        topMargin=19 * mm,
        bottomMargin=23 * mm,
        title="KV Cache 최적화 기술 평가",
        author="SKALA LangGraph",
    )
    doc.build(
        _markdown_to_flowables(markdown_text, _styles(font)),
        onFirstPage=_draw_footer(font),
        onLaterPages=_draw_footer(font),
    )
    return output
