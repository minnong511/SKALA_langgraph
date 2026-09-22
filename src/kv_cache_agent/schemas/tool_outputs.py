"""검색·원문 수집 도구의 정규화된 반환 형식."""

from typing import Literal, TypedDict


class WebSearchResult(TypedDict, total=False):
    """Tavily 검색 결과에서 에이전트에 필요한 필드만 보존한 형식."""

    result_id: str
    title: str
    url: str
    content: str
    score: float
    raw_content: str | None
    published_date: str
    favicon: str
    topic: Literal["general", "news", "finance"]
    source_type: Literal["web"]
    retrieval_method: Literal["tavily"]


class FetchedSource(TypedDict, total=False):
    """웹 페이지 또는 PDF 원문 수집 결과 형식."""

    title: str
    url: str
    content: str
    source_type: Literal["web", "pdf"]
    published_date: str
    content_length: int
    status_code: int
    content_type: str
    fetch_status: Literal["ok", "empty", "unsupported", "error"]
    error: str
