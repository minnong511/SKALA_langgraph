"""Render Markdown to an embedded-font Korean PDF and measure its SUMMARY."""

from __future__ import annotations

import hashlib
import os
import re
import threading
from html import escape
from io import BytesIO
from pathlib import Path
from typing import Any

from src.common.artifacts import atomic_write_exclusive

_FONT_LOCK = threading.Lock()


def _font_candidates(font_path: Path | None) -> list[Path]:
    if font_path is not None:
        return [Path(font_path).expanduser()]
    configured = os.environ.get("PDF_FONT_PATH")
    if configured:
        return [Path(configured).expanduser()]
    return [
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf"),
        Path("/usr/share/fonts/truetype/nanum/NanumMyeongjo.ttf"),
        Path("/usr/local/share/fonts/NanumGothic.ttf"),
        Path.home() / ".fonts/NanumGothic.ttf",
        Path.home() / ".local/share/fonts/NanumGothic.ttf",
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/Library/Fonts/NanumGothic.ttf"),
        Path.home() / "Library/Fonts/NanumGothic.ttf",
    ]


def _register_font(font_path: Path | None, markdown: str) -> tuple[str, Path]:
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    required = {
        ord(character)
        for character in markdown
        if (
            "\uac00" <= character <= "\ud7a3"
            or "\u3130" <= character <= "\u318f"
            or "\u1100" <= character <= "\u11ff"
        )
    } | {ord("한"), ord("글")}
    failures = []
    for candidate in _font_candidates(font_path):
        if not candidate.is_file():
            continue
        try:
            name = "KoreanReport_" + hashlib.sha256(str(candidate.resolve()).encode()).hexdigest()[:12]
            with _FONT_LOCK:
                if name not in pdfmetrics.getRegisteredFontNames():
                    font = TTFont(name, str(candidate))
                    pdfmetrics.registerFont(font)
                    pdfmetrics.registerFontFamily(name, normal=name, bold=name, italic=name, boldItalic=name)
                font = pdfmetrics.getFont(name)
            missing = required.difference(font.face.charToGlyph)
            if missing:
                failures.append(f"{candidate}: missing {len(missing)} Korean characters")
                continue
            return name, candidate
        except Exception as error:
            failures.append(f"{candidate}: {error}")
    reason = "; ".join(failures) or "No Korean TrueType font was found"
    raise RuntimeError(f"{reason}. Install Nanum fonts or set PDF_FONT_PATH to a Korean .ttf font.")


def _inline(text: str) -> str:
    """Escape untrusted source text before applying a small Markdown subset."""
    text = escape(text, quote=True)
    text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<link href="\2" color="#24538a">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text or "&#160;"


def export_pdf(markdown: str, destination: Path, *, font_path: Path | None = None) -> dict[str, Any]:
    """Save an immutable PDF after successful rendering, returning measured layout.

    SUMMARY extends to the next heading at the same or higher level. Its occupied
    height includes its heading, spacing, and page transitions, as laid out by
    ReportLab. The half-page allowance is half the usable A4 page height.
    """
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.platypus import (
        BaseDocTemplate,
        Flowable,
        Frame,
        PageTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )

    if not isinstance(markdown, str) or not markdown.strip():
        raise ValueError("Cannot export an empty report")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    font_name, resolved_font = _register_font(font_path, markdown)
    page_width, page_height = A4
    margin = 48
    body_width = page_width - margin * 2
    body_height = page_height - margin * 2
    body_style = ParagraphStyle(
        "Body",
        fontName=font_name,
        fontSize=10,
        leading=15,
        spaceAfter=7,
        wordWrap="CJK",
        alignment=TA_LEFT,
        splitLongWords=True,
        allowWidows=0,
        allowOrphans=0,
    )
    heading_styles = {
        level: ParagraphStyle(
            f"Heading{level}",
            parent=body_style,
            fontSize=max(11, 19 - level * 2),
            leading=max(17, 26 - level * 2),
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=True,
            textColor=colors.HexColor("#18334b"),
        )
        for level in range(1, 7)
    }
    list_style = ParagraphStyle("List", parent=body_style, leftIndent=12, firstLineIndent=-8, spaceAfter=4)
    table_style = ParagraphStyle("Cell", parent=body_style, fontSize=8.5, leading=12, spaceAfter=0)
    code_style = ParagraphStyle(
        "Code",
        parent=body_style,
        fontSize=9,
        leading=13,
        backColor=colors.HexColor("#f3f5f7"),
        leftIndent=8,
        rightIndent=8,
    )

    class SummaryMarker(Flowable):
        def __init__(self, boundary: str):
            super().__init__()
            self.boundary = boundary
            self.width = self.height = 0
            self.keepWithNext = boundary == "start"

        def draw(self):
            pass

    class MeasuredDocument(BaseDocTemplate):
        def __init__(self, *args: Any, **kwargs: Any):
            super().__init__(*args, **kwargs)
            self.summary_bounds: dict[str, tuple[int, float]] = {}
            self.rendered_pages = 0

        def afterFlowable(self, flowable: Any):
            self.rendered_pages = max(self.rendered_pages, self.page)
            if isinstance(flowable, SummaryMarker):
                self.summary_bounds[flowable.boundary] = (self.page, self.frame._y)

    def page_footer(canvas: Any, document: Any):
        canvas.saveState()
        canvas.setFont(font_name, 8)
        canvas.setFillColor(colors.HexColor("#68717b"))
        canvas.drawRightString(page_width - margin, margin / 2, str(document.page))
        canvas.restoreState()

    stream = BytesIO()
    document = MeasuredDocument(
        stream,
        pagesize=A4,
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
        title="기술 평가 보고서",
        author="SKALA research workflow",
    )
    frame = Frame(
        margin, margin, body_width, body_height, leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0
    )
    document.addPageTemplates(PageTemplate(id="report", frames=[frame], onPage=page_footer))
    story: list[Any] = []
    lines = markdown.splitlines()
    index = 0
    summary_level: int | None = None
    summary_seen = False
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level, text = len(heading[1]), heading[2]
            if summary_level is not None and level <= summary_level:
                story.append(SummaryMarker("end"))
                summary_level = None
            normalized = re.sub(r"^\d+[.)]?\s*", "", text).strip().casefold()
            if not summary_seen and (
                normalized in {"summary", "executive summary", "요약", "요약문"}
                or normalized.startswith("summary (")
            ):
                summary_seen = True
                summary_level = level
                story.append(SummaryMarker("start"))
            story.append(Paragraph(_inline(text), heading_styles[level]))
            index += 1
            continue
        if line.startswith("```"):
            code_lines = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(escape(lines[index]).replace(" ", "&#160;"))
                index += 1
            story.append(Paragraph("<br/>".join(code_lines) or "&#160;", code_style))
            index += 1
            continue
        if (
            "|" in line
            and index + 1 < len(lines)
            and re.fullmatch(r"[\s|:\-]+", lines[index + 1])
            and "-" in lines[index + 1]
        ):

            def cells(row: str) -> list[str]:
                return [part.strip() for part in row.strip().strip("|").split("|")]

            rows = [cells(line)]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                rows.append(cells(lines[index]))
                index += 1
            columns = max(len(row) for row in rows)
            data = [
                [Paragraph(_inline(cell), table_style) for cell in row + [""] * (columns - len(row))]
                for row in rows
            ]
            table = Table(
                data,
                colWidths=[body_width / columns] * columns,
                repeatRows=1,
                splitByRow=1,
                splitInRow=1,
                hAlign="LEFT",
            )
            table.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e9eef3")),
                        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cdd5dd")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                        ("TOPPADDING", (0, 0), (-1, -1), 5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ]
                )
            )
            story.extend([table, Spacer(1, 7)])
            continue
        bullet = re.match(r"^(?:[-*+]\s+|\d+[.)]\s+)(.+)$", line)
        if bullet:
            number = re.match(r"^(\d+[.)])", line)
            prefix = number[1] if number else "•"
            story.append(Paragraph(f"{prefix} {_inline(bullet[1])}", list_style))
            index += 1
            continue
        if re.fullmatch(r"[-*_]{3,}", line):
            story.append(Spacer(1, 7))
            index += 1
            continue
        paragraph_lines = [line.lstrip("> ") if line.startswith(">") else line]
        index += 1
        while index < len(lines):
            following = lines[index].strip()
            if (
                not following
                or re.match(r"^(#{1,6}\s|[-*+]\s|\d+[.)]\s|```|>)", following)
                or "|" in following
            ):
                break
            paragraph_lines.append(following)
            index += 1
        story.append(Paragraph(_inline(" ".join(paragraph_lines)), body_style))
    if summary_level is not None:
        story.append(SummaryMarker("end"))
    document.build(story)
    start = document.summary_bounds.get("start")
    end = document.summary_bounds.get("end")
    summary_height = 0.0
    if start is not None and end is not None:
        summary_height = (end[0] - start[0]) * body_height + start[1] - end[1]
    destination.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_exclusive(destination, stream.getvalue())
    return {
        "path": str(destination),
        "pages": document.rendered_pages,
        "font_path": str(resolved_font),
        "summary_present": summary_seen,
        "summary_height_points": round(summary_height, 2),
        "summary_limit_points": round(body_height / 2, 2),
        "summary_within_half_page": summary_seen
        and start is not None
        and end is not None
        and summary_height <= body_height / 2,
    }
