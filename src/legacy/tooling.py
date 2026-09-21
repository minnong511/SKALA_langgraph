"""Adapters for injected RAG retrievers and web-search tools."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any


def retrieve_context(
    retriever: Any,
    query: str,
    *,
    agent_name: str,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    if retriever is None:
        return []

    if callable(retriever) and not hasattr(retriever, "invoke"):
        raw = retriever(query, top_k)
    else:
        raw = retriever.invoke(query)

    if isinstance(raw, Mapping):
        raw = raw.get("documents") or raw.get("results") or [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = [raw]

    results: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw)[:top_k], start=1):
        if hasattr(item, "page_content"):
            content = str(item.page_content)
            metadata = dict(getattr(item, "metadata", {}) or {})
        elif isinstance(item, Mapping):
            metadata = dict(item.get("metadata", {}) or {})
            content = str(
                item.get("page_content")
                or item.get("content")
                or item.get("text")
                or ""
            )
            metadata = {**dict(item), **metadata}
        else:
            content = str(item)
            metadata = {}

        source = str(metadata.get("source") or metadata.get("url") or "local-rag")
        source_id = str(
            metadata.get("source_id")
            or _stable_id("RAG", agent_name, source, str(index))
        )
        results.append(
            {
                "source_id": source_id,
                "source_type": metadata.get("source_type", "paper"),
                "title": metadata.get("title") or metadata.get("file_name") or source,
                "url": metadata.get("url", ""),
                "published_at": metadata.get("published_at", ""),
                "page_or_section": _page_label(metadata),
                "content": content[:5000],
                "query": query,
                "retrieval_method": "FAISS",
            }
        )
    return results


def search_web(
    web_search: Any,
    query: str,
    *,
    agent_name: str,
    max_results: int = 5,
) -> list[dict[str, Any]]:
    if web_search is None:
        return []

    if callable(web_search) and not hasattr(web_search, "invoke"):
        raw = web_search(query, max_results)
    else:
        try:
            raw = web_search.invoke({"query": query})
        except (TypeError, ValueError):
            raw = web_search.invoke(query)

    if isinstance(raw, Mapping):
        raw = raw.get("results") or raw.get("data") or [raw]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = [raw]

    results: list[dict[str, Any]] = []
    for index, item in enumerate(list(raw)[:max_results], start=1):
        if isinstance(item, Mapping):
            content = str(
                item.get("raw_content")
                or item.get("content")
                or item.get("snippet")
                or ""
            )
            url = str(item.get("url") or "")
            title = str(item.get("title") or url or f"web-result-{index}")
            published_at = str(
                item.get("published_at") or item.get("published_date") or ""
            )
        else:
            content = str(item)
            url = ""
            title = f"web-result-{index}"
            published_at = ""

        explicit_source_id = ""
        if isinstance(item, Mapping):
            explicit_source_id = str(item.get("source_id") or "")
        results.append(
            {
                "source_id": explicit_source_id
                or _stable_id("WEB", agent_name, url or title, str(index)),
                "source_type": "web",
                "title": title,
                "url": url,
                "published_at": published_at,
                "page_or_section": "",
                "content": content[:5000],
                "query": query,
                "retrieval_method": "Tavily",
            }
        )
    return results


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}-{digest}"


def _page_label(metadata: Mapping[str, Any]) -> str:
    if metadata.get("page_label") is not None:
        return str(metadata["page_label"])
    if metadata.get("page") is not None:
        # Most PDF loaders store zero-based page indexes.
        try:
            return f"p.{int(metadata['page']) + 1}"
        except (TypeError, ValueError):
            return str(metadata["page"])
    return str(metadata.get("section") or "")
