"""Markdown 보고서의 한국어 PDF 변환 테스트."""

from pathlib import Path

import pytest
from pypdf import PdfReader

from kv_cache_agent.tools.pdf_writer import (
    ReportPageLimitError,
    _find_korean_font,
    pdf_page_count,
    write_pdf,
)


def test_write_pdf_with_korean_markdown(tmp_path: Path):
    """한국어 제목·본문과 페이지 번호가 포함된 PDF를 생성하는지 확인."""
    try:
        _find_korean_font()
    except RuntimeError as error:
        pytest.skip(str(error))

    output = tmp_path / "report.pdf"
    markdown = """# SUMMARY

TurboQuant와 CXL-based 기술을 클라우드 LLM 서빙 관점에서 비교한다. [1]

## 1. 분석 배경

해석: 부분 검증 근거를 포함한 잠정 평가다.

# REFERENCE

[1] 테스트 자료. 근거 ID: test-001
"""

    result = write_pdf(markdown, output)

    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0
    assert len(PdfReader(str(output)).pages) == 1


def test_page_limit_includes_references_and_preserves_existing_file(tmp_path):
    markdown = "# SUMMARY\n\n짧은 본문.\n\n# REFERENCE\n\n" + "\n\n".join(
        f"[{i}] " + "원문과 근거 위치를 보존하는 참고문헌. " * 12
        for i in range(100)
    )
    assert pdf_page_count(markdown) > 10
    output = tmp_path / "report.pdf"
    output.write_bytes(b"previous artifact")
    with pytest.raises(ReportPageLimitError):
        write_pdf(markdown, output)
    assert output.read_bytes() == b"previous artifact"


def test_reference_source_and_evidence_id_stay_on_same_page(tmp_path):
    refs = [f"[{i}] " + "근거 원문과 위치. " * 15 + f" 근거 ID: source-{i}" for i in range(55)]
    output = write_pdf("# SUMMARY\n\n평가.\n\n# REFERENCE\n\n" + "\n\n".join(refs), tmp_path / "refs.pdf")
    texts = [page.extract_text() for page in PdfReader(output).pages]
    assert any("REFERENCE" in text and "[0]" in text for text in texts)
    for i in range(55):
        assert any(f"[{i}]" in text and f"source-{i}" in text for text in texts)
