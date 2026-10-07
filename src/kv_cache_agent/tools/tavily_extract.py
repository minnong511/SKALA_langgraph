"""Tavily URL extraction with per-URL outcomes and immutable source snapshots."""

import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any
from urllib.parse import urlparse, urlunparse

from tavily import TavilyClient

from kv_cache_agent.config import (
    DATA_DIR,
    TAVILY_API_KEY,
    TAVILY_EXTRACT_DEPTH,
    TAVILY_EXTRACT_TIMEOUT,
)
from kv_cache_agent.observability.logger import emit
from kv_cache_agent.observability.tracing import traced_tool
from kv_cache_agent.schemas.base import Contract, stable_id
from kv_cache_agent.schemas.evidence import SourceRef, SourceSnapshot
from kv_cache_agent.schemas.research import BudgetLedger


def canonical_url(url: str) -> str:
    parsed = urlparse(url.strip())
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError("A public HTTP(S) URL without credentials is required")
    return urlunparse(
        parsed._replace(
            fragment="", scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower()
        )
    )


class ExtractBatch(Contract):
    snapshots: tuple[SourceSnapshot, ...]
    credits: float | None = None
    request_ids: tuple[str, ...] = ()
    cache_hits: int = 0


class SourceCache:
    def __init__(self, directory: Path, *, ttl_seconds: float = 86400):
        if ttl_seconds < 0:
            raise ValueError("Cache TTL must be nonnegative")
        self.directory = Path(directory)
        self.ttl = timedelta(seconds=ttl_seconds)
        self._lock = RLock()

    def _path(self, url: str) -> Path:
        return self.directory / f"{stable_id('url', canonical_url(url))}.json"

    def get(self, url: str, now: datetime) -> SourceSnapshot | None:
        with self._lock:
            path = self._path(url)
            if not path.is_file():
                return None
            try:
                snapshot = SourceSnapshot.model_validate_json(path.read_text())
            except (ValueError, OSError):
                return None
            age = now - snapshot.retrieved_at
            if snapshot.status != "ok" or age < timedelta(0) or age >= self.ttl:
                return None
            return snapshot

    def put(self, snapshot: SourceSnapshot):
        if snapshot.status != "ok":
            return
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self._path(snapshot.reference.location)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(snapshot.model_dump_json(), encoding="utf-8")
            temporary.replace(path)


class TavilyExtractor:
    def __init__(
        self,
        *,
        client: Any = None,
        cache: SourceCache | None = None,
        budget: BudgetLedger | None = None,
        depth: str = "advanced",
        timeout: float = 30,
        batch_size: int = 20,
        clock=None,
    ):
        if (
            depth not in {"basic", "advanced"}
            or not 1 <= timeout <= 60
            or not 1 <= batch_size <= 20
        ):
            raise ValueError("Invalid Tavily extraction options")
        self.client = client
        self.cache = cache
        self.budget = budget
        self.depth = depth
        self.timeout = timeout
        self.batch_size = batch_size
        self.clock = clock or (lambda: datetime.now(UTC))
        # Single-flight per service instance, including parallel graph nodes.
        self._lock = RLock()

    def _client(self):
        if self.client is None:
            key = os.getenv("TAVILY_API_KEY") or TAVILY_API_KEY
            if not key:
                raise ValueError("TAVILY_API_KEY is required for extraction")
            self.client = TavilyClient(api_key=key)
        return self.client

    @traced_tool("tavily.extract")
    def extract(self, urls: list[str]) -> ExtractBatch:
        with self._lock:
            return self._extract(urls)

    def _extract(self, urls: list[str]) -> ExtractBatch:
        requested = list(dict.fromkeys(canonical_url(url) for url in urls))
        now = self.clock()
        results: dict[str, SourceSnapshot] = {}
        pending: list[str] = []
        for url in requested:
            cached = self.cache.get(url, now) if self.cache else None
            if cached is not None:
                results[url] = cached
                emit(
                    "source_cache_hit",
                    details={"url": url, "content_hash": cached.content_hash},
                )
            else:
                pending.append(url)
        hits = len(results)
        ids: list[str] = []
        credits: float | None = None
        for start in range(0, len(pending), self.batch_size):
            batch = pending[start : start + self.batch_size]
            if self.budget:
                self.budget.reserve(extract_calls=1, extract_urls=len(batch))
            emit(
                "tool_start",
                message="Tavily 본문 추출",
                details={"urls": batch, "depth": self.depth},
            )
            try:
                # Do not send query: it would return short reranked chunks, not the body.
                response = self._client().extract(
                    urls=batch,
                    extract_depth=self.depth,
                    format="markdown",
                    timeout=self.timeout,
                    include_usage=True,
                )
                if not isinstance(response, dict):
                    raise TypeError("Invalid Tavily Extract response")
            except Exception as error:  # noqa: BLE001 - external boundary returns explicit failure
                response = {
                    "failed_results": [
                        {"url": url, "error": f"{type(error).__name__}: {error}"}
                        for url in batch
                    ]
                }
                emit(
                    "tool_error",
                    level="WARNING",
                    message=type(error).__name__,
                    details={"url_count": len(batch)},
                )
            request_id = response.get("request_id")
            if request_id:
                ids.append(str(request_id))
            usage = (
                response.get("usage", {}).get("credits")
                if isinstance(response.get("usage"), dict)
                else None
            )
            if isinstance(usage, (int, float)):
                credits = (credits or 0) + usage
            successful: dict[str, dict] = {}
            failures: dict[str, str] = {}
            for item in response.get("results", []):
                if isinstance(item, dict):
                    try:
                        key = canonical_url(str(item.get("url", "")))
                    except ValueError:
                        continue
                    if key in batch:
                        successful[key] = item
            for item in response.get("failed_results", []):
                if isinstance(item, dict):
                    try:
                        key = canonical_url(str(item.get("url", "")))
                    except ValueError:
                        continue
                    failures[key] = str(item.get("error", "Extraction failed"))
            for url in batch:
                item = successful.get(url, {})
                content = item.get("raw_content", "")
                content = content.strip() if isinstance(content, str) else ""
                error = failures.get(url, "")
                status = (
                    "ok"
                    if content and not error
                    else "blocked"
                    if any(code in error for code in ("401", "403", "451"))
                    else "error"
                    if error
                    else "empty"
                    if item
                    else "error"
                )
                if status != "ok":
                    content = ""
                    error = error or "Tavily did not return source content"
                snapshot = SourceSnapshot(
                    reference=SourceRef(
                        source_id=stable_id("source", url),
                        location=url,
                        title=str(item.get("title", "")),
                        locator="extracted body",
                    ),
                    content=content,
                    acquisition="tavily_extract",
                    status=status,
                    retrieved_at=now,
                    error=error,
                    provider_request_id=str(request_id) if request_id else None,
                )
                results[url] = snapshot
                if self.cache:
                    self.cache.put(snapshot)
            emit(
                "tool_end",
                details={
                    "urls": batch,
                    "success": sum(results[url].status == "ok" for url in batch),
                    "failed": sum(results[url].status != "ok" for url in batch),
                    "credits": usage,
                },
            )
        return ExtractBatch(
            snapshots=tuple(results[url] for url in requested),
            credits=credits,
            request_ids=tuple(ids),
            cache_hits=hits,
        )


_DEFAULT_EXTRACTOR: TavilyExtractor | None = None
_DEFAULT_LOCK = RLock()


def get_extractor() -> TavilyExtractor:
    global _DEFAULT_EXTRACTOR
    with _DEFAULT_LOCK:
        if _DEFAULT_EXTRACTOR is None:
            _DEFAULT_EXTRACTOR = TavilyExtractor(
                cache=SourceCache(DATA_DIR / "cache" / "sources"),
                depth=TAVILY_EXTRACT_DEPTH,
                timeout=TAVILY_EXTRACT_TIMEOUT,
            )
        return _DEFAULT_EXTRACTOR


def web_fetch(url: str, *, extractor: TavilyExtractor | None = None) -> SourceSnapshot:
    return (extractor or get_extractor()).extract([url]).snapshots[0]
