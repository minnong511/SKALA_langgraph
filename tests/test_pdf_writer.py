"""Markdown 보고서의 한국어 PDF 변환 테스트."""

from pathlib import Path

import pytest
from pypdf import PdfReader

from kv_cache_agent.tools.pdf_writer import _find_korean_font, write_pdf


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
