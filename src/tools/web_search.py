"""Tavily search with explicit configuration and stable source provenance.

API reference: https://docs.tavily.com/documentation/api-reference/endpoint/search
"""

from __future__ import annotations

import hashlib
import math
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from src.schemas import SourceDocument


class SearchError(RuntimeError):
    """The search service could not return a valid result."""


def canonical_url(url: str) -> str:
    """Ignore fragments and tracking parameters without dropping meaningful queries."""
    parsed = urlsplit(url.strip())
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        raise ValueError("A source URL must use http or https and include a host")
    if parsed.username or parsed.password:
        raise ValueError("Source URLs cannot contain credentials")
    host = parsed.hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port and not (
        (parsed.scheme.lower() == "https" and port == 443) or (parsed.scheme.lower() == "http" and port == 80)
    ):
        host += f":{port}"
    query = [
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in {"fbclid", "gclid"}
    ]
    return urlunsplit((parsed.scheme.lower(), host, parsed.path or "/", urlencode(sorted(query)), ""))


def stable_source_id(prefix: str, identity: str) -> str:
    return f"{prefix}-{hashlib.sha256(identity.encode('utf-8')).hexdigest()[:20]}"


class TavilySearch:
    """HTTP client may be injected for deterministic, network-free tests."""

    def __init__(
        self,
        api_key: str,
        *,
        timeout_seconds: float = 20.0,
        client: Any = None,
        search_depth: str = "advanced",
    ) -> None:
        if not api_key or not api_key.strip():
            raise ValueError("TAVILY_API_KEY is required for web search")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if search_depth not in {"basic", "advanced", "fast", "ultra-fast"}:
            raise ValueError("Unsupported Tavily search depth")
        self._api_key = api_key
        self.timeout_seconds = timeout_seconds
        self.search_depth = search_depth
        self._client = client

    def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 20:
            raise ValueError("Tavily limit must be an integer between 1 and 20")
        payload = {
            "query": query.strip(),
            "max_results": limit,
            "search_depth": self.search_depth,
            "topic": "general",
            "include_answer": False,
            "include_raw_content": "text",
            "include_published_date": True,
        }
        client = self._client or httpx.Client(follow_redirects=False)
        try:
            response = client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            # Do not copy request headers or the API key into persisted diagnostics.
            raise SearchError(f"Tavily search failed ({type(exc).__name__})") from exc
        finally:
            if self._client is None:
                client.close()
        if not isinstance(data, dict) or not isinstance(data.get("results"), list):
            raise SearchError("Tavily returned an invalid results payload")
        accessed_at = datetime.now(UTC).isoformat()
        sources: list[SourceDocument] = []
        seen: set[str] = set()
        for item in data["results"]:
            if not isinstance(item, dict):
                continue
            try:
                url = canonical_url(str(item.get("url") or ""))
            except ValueError:
                continue
            if url in seen:
                continue
            content = str(item.get("raw_content") or item.get("content") or "").strip()
            if not content:
                continue
            seen.add(url)
            sources.append(
                SourceDocument(
                    source_id=stable_source_id("WEB", url),
                    source_type="web",
                    title=str(item.get("title") or url),
                    url=str(item["url"]),
                    content=content,
                    published_at=str(item.get("published_date") or item.get("published_at") or ""),
                    accessed_at=accessed_at,
                    page_or_section="",
                    query=query.strip(),
                    retrieval_method="Tavily",
                    metadata={
                        "score": item.get("score"),
                        "original_url": item["url"],
                        "content_kind": "raw_content" if item.get("raw_content") else "search_excerpt",
                        "publication_date_missing_reason": ""
                        if item.get("published_date") or item.get("published_at")
                        else "검색 서비스가 발행일을 제공하지 않음",
                    },
                )
            )
            if len(sources) == limit:
                break
        return sources


WebSearch = TavilySearch
