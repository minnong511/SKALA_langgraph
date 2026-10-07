"""Shared original-source collection for bounded worker assignments."""

import re
from collections.abc import Callable
from pathlib import Path
from threading import RLock

import httpx

from kv_cache_agent.config import ROOT_DIR
from kv_cache_agent.observability.logger import emit
from kv_cache_agent.schemas.base import Contract, stable_id
from kv_cache_agent.schemas.evidence import SourceRef, SourceSnapshot
from kv_cache_agent.schemas.research import BudgetExceeded, BudgetLedger
from kv_cache_agent.schemas.worker import WorkerInput
from kv_cache_agent.tools.paper_retriever import retrieve_paper_chunks
from kv_cache_agent.tools.tavily_extract import canonical_url, get_extractor
from kv_cache_agent.tools.tavily_search import search_web
from kv_cache_agent.verification.sources import (
    SourceLoader,
    classify_source,
    parse_pages,
    reference_key,
)

MAX_CONTEXT_CHARACTERS = 60_000
MAX_SOURCE_CHARACTERS = 6_000
MAX_SOURCE_WINDOWS = 2
MAX_WEB_URLS = 6


class TaskBudget:
    """Per-task counters with reservations against one shared run ledger."""

    def __init__(self, request: WorkerInput, ledger: BudgetLedger):
        self.task = request.task
        self.ledger = ledger
        self.used = dict.fromkeys(ledger.snapshot(), 0)
        self.used["paper_queries"] = 0
        self._lock = RLock()

    def reserve(self, **amounts):
        with self._lock:
            for key, maximum in (
                ("search_calls", self.task.max_search_calls),
                ("model_calls", self.task.max_model_calls),
                ("extract_calls", self.task.max_extract_calls),
                ("extract_urls", self.task.max_extract_urls),
            ):
                if self.used[key] + amounts.get(key, 0) > maximum:
                    raise BudgetExceeded(f"Task budget exhausted: {key}")
            self.ledger.reserve(**amounts)
            for key, amount in amounts.items():
                self.used[key] += amount

    def snapshot(self):
        with self._lock:
            return dict(self.used)


class SourceCollection(Contract):
    snapshots: tuple[SourceSnapshot, ...] = ()
    limitations: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


def build_queries(request: WorkerInput) -> list[str]:
    """Round-robin technologies; query text never becomes source evidence."""
    task = request.task
    limit = min(task.max_search_calls, 3)
    if task.agent == "technical":
        # A local retrieval has no Tavily charge, but remains a bounded operation.
        limit = max(1, limit)
    queries = []
    for index in range(limit):
        technology = task.technologies[index % len(task.technologies)]
        criterion = task.criteria[
            (index // len(task.technologies)) % len(task.criteria)
        ]
        question = task.questions[index % len(task.questions)] if task.questions else ""
        queries.append(
            f"{technology} KV cache {task.agent} {criterion} {question}".strip()
        )
    return list(dict.fromkeys(queries))


def _normalize(text: str) -> str:
    return " ".join(text.split()).casefold()


def source_passages(snapshot: SourceSnapshot) -> tuple[dict[str, str], ...]:
    """Expose exact original passages so the model selects IDs instead of rewriting quotes."""
    passages = []
    for paragraph in re.finditer(
        r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", snapshot.content, re.DOTALL
    ):
        for start in range(paragraph.start(), paragraph.end(), 1000):
            end = min(start + 1000, paragraph.end())
            text = snapshot.content[start:end]
            if not text.strip():
                continue
            passages.append(
                {
                    "passage_id": stable_id(
                        "passage",
                        (
                            snapshot.reference.source_id,
                            snapshot.content_hash,
                            start,
                            end,
                        ),
                    ),
                    "text": text,
                }
            )
    return tuple(passages)


def select_windows(snapshot: SourceSnapshot, request: WorkerInput):
    """Keep bounded, versioned body spans; never pretend a prefix is the full body."""
    if len(snapshot.content) <= MAX_SOURCE_CHARACTERS:
        return [snapshot]
    words = set(
        re.findall(
            r"[\w-]{3,}",
            " ".join(
                (
                    *request.task.technologies,
                    *request.task.criteria,
                    *request.task.questions,
                )
            ).lower(),
        )
    )
    candidates = []
    for start in range(0, len(snapshot.content), MAX_SOURCE_CHARACTERS):
        end = min(start + MAX_SOURCE_CHARACTERS, len(snapshot.content))
        body = snapshot.content[start:end].lower()
        score = sum(min(body.count(word), 5) for word in words)
        candidates.append((score, start, end))
    selected = sorted(candidates, key=lambda item: (-item[0], item[1]))[
        :MAX_SOURCE_WINDOWS
    ]
    windows = []
    for _, start, end in sorted(selected, key=lambda item: item[1]):
        reference = snapshot.reference
        offset = reference.character_range[0] if reference.character_range else 0
        version = (
            reference.version
            if reference.character_range or snapshot.acquisition == "local_pdf"
            else snapshot.content_hash
        )
        if not version:
            raise ValueError("Source windows require a document version")
        span = (offset + start, offset + end)
        reference = reference.model_copy(
            update={
                "source_id": stable_id(
                    "source", (reference_key(reference), version, span)
                ),
                "version": version,
                "character_range": span,
                "locator": reference.locator
                if snapshot.acquisition == "local_pdf"
                else f"body characters {span[0]}:{span[1]}",
            }
        )
        windows.append(
            SourceSnapshot(
                reference=reference,
                content=snapshot.content[start:end],
                acquisition=snapshot.acquisition,
                status="ok",
                retrieved_at=snapshot.retrieved_at,
                provider_request_id=snapshot.provider_request_id,
            )
        )
    emit(
        "source_windows_selected",
        details={
            "location": snapshot.reference.location,
            "body_characters": len(snapshot.content),
            "spans": [s.reference.character_range for s in windows],
            "body_version": version,
        },
    )
    return windows


def _transient(error: Exception) -> bool:
    cause = error.__cause__ or error
    if isinstance(cause, (httpx.TimeoutException, httpx.ConnectError, TimeoutError)):
        return True
    response = getattr(cause, "response", None)
    return getattr(response, "status_code", None) in {429, 500, 502, 503, 504}


class SourceCollector:
    def __init__(
        self,
        *,
        root: Path = ROOT_DIR,
        search: Callable = search_web,
        paper_search: Callable = retrieve_paper_chunks,
        extractor=None,
        loader: Callable | None = None,
    ):
        self.root = Path(root).resolve()
        self.search = search
        self.paper_search = paper_search
        self.extractor = extractor
        self.loader = loader

    def collect(self, request: WorkerInput, budget: TaskBudget) -> SourceCollection:
        extractor = self.extractor or get_extractor()
        load = (
            self.loader
            or SourceLoader(root=self.root, extractor=extractor, budget=budget).load
        )
        snapshots = []
        limitations = []
        errors = []
        identities = set()
        characters = 0

        def bounded_urls(urls):
            remaining = request.task.max_extract_urls - budget.used["extract_urls"]
            selected = []
            for url in urls:
                cached = (
                    extractor.cache.get(url, extractor.clock())
                    if extractor.cache
                    else None
                )
                if cached is not None:
                    selected.append(url)
                elif remaining > 0:
                    selected.append(url)
                    remaining -= 1
                else:
                    limitations.append(f"Extraction URL budget cannot fetch {url}")
            return selected

        def add(snapshot):
            nonlocal characters
            if (
                snapshot.status != "ok"
                or snapshot.truncated
                or (snapshot.acquisition == "search_excerpt")
            ):
                limitations.append(
                    f"Original unavailable: {snapshot.reference.location} "
                    f"({snapshot.status}; {snapshot.error})"
                )
                return
            snapshot = snapshot.model_copy(
                update={
                    "reference": snapshot.reference.model_copy(
                        update={
                            "source_type": classify_source(snapshot.reference.location)
                        }
                    )
                }
            )
            try:
                selected = select_windows(snapshot, request)
            except ValueError as error:
                limitations.append(str(error))
                return
            for item in selected:
                key = (reference_key(item.reference), item.reference.version)
                if key in identities:
                    continue
                if characters + len(item.content) > MAX_CONTEXT_CHARACTERS:
                    limitations.append(
                        "Source context limit reached; narrow the assignment"
                    )
                    continue
                identities.add(key)
                item = item.model_copy(
                    update={
                        "reference": item.reference.model_copy(
                            update={"source_id": stable_id("source", key)}
                        )
                    }
                )
                snapshots.append(item)
                characters += len(item.content)

        refs = list(request.source_refs)
        for card in request.existing_evidence:
            refs.extend(card.source_refs)
        for result in request.technical_results:
            for card in result.evidence:
                if card.technology in request.task.technologies:
                    refs.extend(card.source_refs)
        failed_prefetch = {}
        if extractor.cache:
            urls = list(
                dict.fromkeys(
                    canonical_url(ref.location)
                    for ref in refs
                    if ref.location.startswith("http")
                )
            )
            selected_urls = bounded_urls(urls)
            if selected_urls:
                try:
                    prefetched = extractor.extract(selected_urls, budget=budget)
                    failed_prefetch = {
                        s.reference.location: s
                        for s in prefetched.snapshots
                        if s.status != "ok"
                    }
                except BudgetExceeded as error:
                    limitations.append(str(error))
        seen_refs = set()
        for ref in refs:
            key = (reference_key(ref), ref.version)
            if key in seen_refs:
                continue
            seen_refs.add(key)
            try:
                if (
                    ref.location.startswith("http")
                    and canonical_url(ref.location) in failed_prefetch
                ):
                    add(failed_prefetch[canonical_url(ref.location)])
                    continue
                actual = load(ref)
                if ref.version and actual.reference.version != ref.version:
                    raise ValueError("Requested source version changed")
                add(actual)
            except Exception as error:  # noqa: BLE001 - external boundary returns task deficits  # explicit deficit; supervisor decides next step
                errors.append(f"Source {ref.location}: {type(error).__name__}: {error}")

        queries = build_queries(request) if request.task.action == "research" else []
        if request.task.agent in {"technical", "cloud_domain"} and queries:
            try:
                budget.used["paper_queries"] += len(queries)
                chunks = self.paper_search(queries, top_k=4)
                for chunk in chunks:
                    try:
                        metadata = chunk.get("metadata", {})
                        location = str(
                            metadata.get("source_path")
                            or metadata.get("source_url")
                            or ""
                        )
                        if not location:
                            raise ValueError("Paper chunk has no source location")
                        pages = (
                            (int(metadata["page"]),)
                            if metadata.get("page")
                            else (parse_pages(str(metadata.get("source_locator", ""))))
                        )
                        if not pages:
                            raise ValueError("Paper chunk has no page locator")
                        ref = SourceRef(
                            source_id=stable_id(
                                "source", (location, pages, chunk["chunk_id"])
                            ),
                            location=location,
                            source_type="paper",
                            title=str(metadata.get("source_title", "")),
                            pages=pages,
                            chunk_ids=(str(chunk["chunk_id"]),),
                            locator="; ".join(f"p. {p}" for p in pages),
                            version=str(metadata.get("source_version", "")),
                        )
                        original = load(ref)
                        body = str(chunk.get("content", "")).strip()
                        if (
                            original.status != "ok"
                            or not body
                            or (_normalize(body) not in _normalize(original.content))
                        ):
                            raise ValueError(
                                "Retrieved chunk does not match its original page"
                            )
                        reference = original.reference
                        start = original.content.find(body)
                        if start >= 0 and original.acquisition == "local_pdf":
                            reference = reference.model_copy(
                                update={"character_range": (start, start + len(body))}
                            )
                        add(
                            SourceSnapshot(
                                reference=reference,
                                content=body,
                                acquisition=original.acquisition,
                                status="ok",
                                retrieved_at=original.retrieved_at,
                            )
                        )
                    except Exception as error:  # noqa: BLE001 - external boundary returns task deficits
                        errors.append(
                            f"Paper source link: {type(error).__name__}: {error}"
                        )
            except Exception as error:  # noqa: BLE001 - external boundary returns task deficits
                errors.append(f"Paper retrieval: {type(error).__name__}: {error}")

        if request.task.agent != "technical" and queries:
            hits = {}
            for query in queries:
                for attempt in range(2):
                    try:
                        budget.reserve(search_calls=1)
                        results = self.search(
                            query, max_results=3, include_raw_content=False
                        )
                        for hit in results:
                            try:
                                url = canonical_url(str(hit.get("url", "")))
                                hits.setdefault(url, hit)
                            except ValueError:
                                limitations.append(
                                    "Search returned an invalid source URL"
                                )
                        break
                    except BudgetExceeded as error:
                        limitations.append(str(error))
                        break
                    except Exception as error:  # noqa: BLE001 - external boundary returns task deficits
                        errors.append(f"Web search: {type(error).__name__}: {error}")
                        if attempt == 0 and _transient(error):
                            emit(
                                "tool_retry",
                                level="WARNING",
                                details={
                                    "tool": "tavily.search",
                                    "attempt": 2,
                                    "reason": type(error).__name__,
                                },
                            )
                            continue
                        break
                if budget.used["search_calls"] >= request.task.max_search_calls:
                    break
            urls = bounded_urls(list(hits)[:MAX_WEB_URLS])
            if urls:
                try:
                    batch = extractor.extract(urls, budget=budget)
                    for snapshot in batch.snapshots:
                        hit = hits.get(canonical_url(snapshot.reference.location), {})
                        ref = snapshot.reference.model_copy(
                            update={
                                "source_type": classify_source(
                                    snapshot.reference.location
                                ),
                                "title": snapshot.reference.title
                                or str(hit.get("title", "")),
                                "version": snapshot.content_hash,
                                "published_date": hit.get("published_date") or None,
                            }
                        )
                        add(snapshot.model_copy(update={"reference": ref}))
                except BudgetExceeded as error:
                    limitations.append(str(error))
                except Exception as error:  # noqa: BLE001 - external boundary returns task deficits
                    errors.append(f"Web extraction: {type(error).__name__}: {error}")
        emit(
            "source_collection_completed",
            details={
                "sources": len(snapshots),
                "characters": characters,
                "limitations": len(limitations),
                "errors": len(errors),
                "usage": budget.snapshot(),
            },
        )
        return SourceCollection(
            snapshots=tuple(snapshots),
            limitations=tuple(dict.fromkeys(limitations)),
            errors=tuple(dict.fromkeys(errors)),
        )
