"""Tavily 웹 검색과 검색 결과 정규화를 담당하는 도구."""

from typing import Any, Literal

from tavily import TavilyClient

from kv_cache_agent.config import TAVILY_API_KEY
from kv_cache_agent.schemas.tool_outputs import WebSearchResult

SearchTopic = Literal["general", "news", "finance"]
SearchDepth = Literal["basic", "advanced", "fast", "ultra-fast"]


class TavilySearchError(RuntimeError):
    """Tavily 검색을 수행할 수 없을 때 발생하는 예외."""


def normalize_search_results(
    response: dict[str, Any],
    topic: SearchTopic = "general",
) -> list[WebSearchResult]:
    """Tavily 원본 응답에서 에이전트가 사용할 검색 결과만 추출한다.

    Tavily 응답 전체에는 질의 정보, 사용량, 요청 ID 등도 포함되지만,
    이후 평가 에이전트가 필요한 것은 출처별 제목·URL·요약·점수이다.
    URL을 기준으로 중복을 제거하여 동일한 출처가 여러 번 평가되는 문제도
    줄인다.
    """
    normalized: list[WebSearchResult] = []
    seen_urls: set[str] = set()

    for item in response.get("results", []):
        if not isinstance(item, dict):
            continue

        url = str(item.get("url", "")).strip()
        if not url or url in seen_urls:
            continue

        seen_urls.add(url)
        normalized.append(
            {
                "result_id": str(item.get("id", "")),
                "title": str(item.get("title", "")),
                "url": url,
                "content": str(item.get("content", "")),
                "score": float(item.get("score", 0.0) or 0.0),
                "raw_content": item.get("raw_content"),
                "published_date": str(item.get("published_date", "") or ""),
                "favicon": str(item.get("favicon", "") or ""),
                "topic": topic,
                "source_type": "web",
                "retrieval_method": "tavily",
            }
        )

    return normalized


def search_web(
    query: str,
    *,
    max_results: int = 5,
    search_depth: SearchDepth = "basic",
    topic: SearchTopic = "general",
    include_raw_content: bool = False,
    client: TavilyClient | None = None,
) -> list[WebSearchResult]:
    """Tavily에서 웹 검색을 수행하고 공통 결과 형식으로 반환한다.

    ``client``를 주입할 수 있도록 하여 테스트에서는 실제 API를 호출하지
    않고도 검색 결과 정규화와 에러 처리를 검증할 수 있게 했다.
    """
    clean_query = query.strip()
    if not clean_query:
        raise ValueError("검색 질의는 비어 있을 수 없습니다.")
    if not 1 <= max_results <= 20:
        raise ValueError("max_results는 1 이상 20 이하이어야 합니다.")

    search_client = client
    if search_client is None:
        if not TAVILY_API_KEY:
            raise TavilySearchError(
                "TAVILY_API_KEY가 설정되지 않았습니다. .env를 확인하세요."
            )
        search_client = TavilyClient(api_key=TAVILY_API_KEY)

    try:
        response = search_client.search(
            query=clean_query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            include_raw_content=include_raw_content,
        )
    except Exception as error:
        raise TavilySearchError(f"Tavily 검색에 실패했습니다: {error}") from error

    return normalize_search_results(response, topic=topic)
