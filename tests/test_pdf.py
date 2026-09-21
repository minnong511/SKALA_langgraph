from pathlib import Path

import pytest
from pypdf import PdfReader

from src.exporters.pdf import _font_candidates, export_pdf


@pytest.fixture
def korean_font():
    fonts = [path for path in _font_candidates(None) if path.is_file()]
    if not fonts:
        pytest.skip("Install Nanum Korean TrueType fonts for rendering tests")
    return fonts[0]


def test_korean_pdf_embeds_readable_text_and_measures_summary(tmp_path, korean_font):
    markdown = """# 기술 평가 보고서

## SUMMARY
기술의 적용 조건과 한계를 확인했습니다. 후속 검증이 필요합니다.

## 1. 분석 배경
한글 원문을 검토합니다. **검증된 사실**과 주장을 구분합니다.

| 기술 | 관점 | 평가 |
| --- | --- | --- |
| TurboQuant | 기술 | 추가 검증 |
| ITME | 시장 | 자료 부족 |

- 출처와 적용 조건을 함께 기록합니다.
- [참고 자료](https://example.org/paper)

## REFERENCE
[E1] 검토한 한국어 논문.
"""
    destination = tmp_path / "report.pdf"
    metadata = export_pdf(markdown, destination, font_path=korean_font)
    reader = PdfReader(destination)
    text = "\n".join(page.extract_text() for page in reader.pages)
    assert "기술 평가 보고서" in text
    assert "자료 부족" in text
    assert "검증된 사실" in text
    assert metadata["pages"] == len(reader.pages)
    assert 0 < metadata["summary_height_points"] < metadata["summary_limit_points"]
    assert metadata["summary_within_half_page"] is True
    fonts = reader.pages[0]["/Resources"]["/Font"]
    assert any(
        "/FontDescriptor" in font.get_object() and "/FontFile2" in font.get_object()["/FontDescriptor"]
        for font in fonts.values()
    )
    assert not list(tmp_path.glob(".pending-*"))


def test_long_summary_uses_actual_wrapping_and_fails_half_page_rule(tmp_path, korean_font):
    long_summary = "\n\n".join(["적용 조건과 검증 한계를 평가합니다. " * 12 for _ in range(16)])
    metadata = export_pdf(
        f"# SUMMARY\n{long_summary}\n\n# 분석 배경\n내용", tmp_path / "long.pdf", font_path=korean_font
    )
    assert metadata["pages"] > 1
    assert metadata["summary_height_points"] > metadata["summary_limit_points"]
    assert metadata["summary_within_half_page"] is False


def test_missing_summary_does_not_pass_layout_requirement(tmp_path, korean_font):
    metadata = export_pdf(
        "# 분석 배경\n검증한 내용입니다.", tmp_path / "missing-summary.pdf", font_path=korean_font
    )
    assert metadata["summary_present"] is False
    assert metadata["summary_within_half_page"] is False


def test_font_failure_is_explicit_and_leaves_no_output(tmp_path):
    destination = tmp_path / "missing.pdf"
    with pytest.raises(RuntimeError, match="PDF_FONT_PATH"):
        export_pdf("# SUMMARY\n한글 보고서", destination, font_path=tmp_path / "missing.ttf")
    assert not destination.exists()


def test_non_korean_font_cannot_silently_produce_missing_glyphs(tmp_path):
    latin_font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not latin_font.is_file():
        pytest.skip("DejaVu font unavailable")
    with pytest.raises(RuntimeError, match="missing .* Korean"):
        export_pdf("# SUMMARY\n한글 보고서", tmp_path / "tofu.pdf", font_path=latin_font)
    assert not (tmp_path / "tofu.pdf").exists()


def test_pdf_never_replaces_existing_output(tmp_path, korean_font):
    destination = tmp_path / "existing.pdf"
    destination.write_bytes(b"existing content")
    with pytest.raises(FileExistsError):
        export_pdf("# SUMMARY\n한글", destination, font_path=korean_font)
    assert destination.read_bytes() == b"existing content"
