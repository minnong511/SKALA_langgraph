from pathlib import Path

import httpx
import pytest

from kv_cache_agent.tools.source_fetcher import fetch_source
from kv_cache_agent.tools.tavily_search import search_web


class FakeTavilyClient:
    """실제 Tavily API를 호출하지 않는 테스트용 클라이언트."""

    def __init__(self, response: dict):
        self.response = response
        self.last_kwargs: dict = {}

    def search(self, **kwargs):
        self.last_kwargs = kwargs
        return self.response


class FakeHttpClient:
    """httpx.Client의 get 메서드만 흉내 내는 테스트용 클라이언트."""

    def __init__(self, response: httpx.Response):
        self.response = response
        self.requested_url = ""

    def get(self, url: str) -> httpx.Response:
        self.requested_url = url
        return self.response


def test_search_web_normalizes_and_deduplicates_results():
    client = FakeTavilyClient(
        {
            "results": [
                {
                    "id": "result-1",
                    "title": "TurboQuant",
                    "url": "https://example.com/turboquant",
                    "content": "KV Cache를 압축한다.",
                    "score": 0.91,
                    "published_date": "2025-01-01",
                },
                {
                    "id": "result-1-duplicate",
                    "title": "중복 결과",
                    "url": "https://example.com/turboquant",
                    "content": "같은 URL이다.",
                    "score": 0.8,
                },
            ]
        }
    )

    results = search_web("TurboQuant KV Cache", client=client)

    assert len(results) == 1
    assert results[0]["url"] == "https://example.com/turboquant"
    assert results[0]["topic"] == "general"
    assert results[0]["source_type"] == "web"
    assert results[0]["retrieval_method"] == "tavily"
    assert client.last_kwargs["max_results"] == 5
    assert client.last_kwargs["exclude_domains"] == [
        "medium.com",
        "towardsai.net",
        "rocketreach.co",
    ]


def test_search_web_rejects_empty_query():
    with pytest.raises(ValueError, match="검색 질의"):
        search_web("   ", client=FakeTavilyClient({"results": []}))


def test_fetch_source_extracts_html_and_removes_script():
    response = httpx.Response(
        200,
        headers={"content-type": "text/html; charset=utf-8"},
        request=httpx.Request("GET", "https://example.com/page"),
        text=(
            "<html><head><title>테스트 문서</title></head>"
            "<body><main>본문 내용<script>비공개 코드</script></main></body></html>"
        ),
    )
    client = FakeHttpClient(response)

    result = fetch_source("https://example.com/page", client=client)

    assert result["fetch_status"] == "ok"
    assert result["source_type"] == "web"
    assert result["title"] == "테스트 문서"
    assert result["content"] == "본문 내용"
    assert "비공개 코드" not in result["content"]


def test_fetch_source_extracts_pdf_text():
    pdf_path = Path(__file__).parents[1] / "data" / "papers" / "turboquant.pdf"
    response = httpx.Response(
        200,
        headers={"content-type": "application/pdf"},
        request=httpx.Request("GET", "https://example.com/turboquant.pdf"),
        content=pdf_path.read_bytes(),
    )

    result = fetch_source(
        "https://example.com/turboquant.pdf",
        client=FakeHttpClient(response),
        max_chars=2_000,
    )

    assert result["fetch_status"] == "ok"
    assert result["source_type"] == "pdf"
    assert result["content_length"] <= 2_000
    assert result["content"]


def test_fetch_source_returns_structured_error_for_invalid_url():
    result = fetch_source("ftp://example.com/file.pdf")

    assert result["fetch_status"] == "error"
    assert result["content"] == ""


def test_fetch_source_marks_forbidden_as_blocked():
    response = httpx.Response(
        403,
        headers={"content-type": "text/html"},
        request=httpx.Request("GET", "https://example.com/blocked"),
    )

    result = fetch_source(
        "https://example.com/blocked",
        client=FakeHttpClient(response),
    )

    assert result["fetch_status"] == "blocked"
    assert result["status_code"] == 403
    assert "차단" in result["error"]
