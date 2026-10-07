"""Read requested PDF pages without conflating pages of the same document."""

import re
from pathlib import Path
from threading import RLock
from urllib.parse import urlparse

from pypdf import PdfReader

from kv_cache_agent.config import ROOT_DIR
from kv_cache_agent.observability.logger import emit
from kv_cache_agent.observability.tracing import traced_tool
from kv_cache_agent.schemas.base import fingerprint, stable_id
from kv_cache_agent.schemas.evidence import SourceRef, SourceSnapshot
from kv_cache_agent.tools.tavily_extract import TavilyExtractor, get_extractor


def parse_pages(locator: str) -> tuple[int, ...]:
    """Accept p. 2; p. 17 and p. 2-4; reject unknown locator syntax."""
    if not locator.strip():
        return ()
    pages: list[int] = []
    for piece in locator.split(";"):
        piece = re.sub(r"^\s*p(?:p)?\.?\s*", "", piece, flags=re.IGNORECASE).strip()
        match = re.fullmatch(r"(\d+)\s*(?:[-–]\s*(\d+))?", piece)
        if match is None:
            raise ValueError("Invalid PDF page locator")
        first = int(match[1])
        last = int(match[2] or first)
        if first < 1 or last < first:
            raise ValueError("Invalid PDF page range")
        pages.extend(range(first, last + 1))
    return tuple(dict.fromkeys(pages))


def read_pdf_pages(
    path: Path, locator: str = "", *, pages: tuple[int, ...] = ()
) -> str:
    reader = PdfReader(str(path))
    selected = pages or parse_pages(locator) or tuple(range(1, len(reader.pages) + 1))
    if any(page < 1 or page > len(reader.pages) for page in selected):
        raise ValueError("PDF locator exceeds document pages")
    return "\n\n".join(
        (reader.pages[page - 1].extract_text() or "").strip() for page in selected
    ).strip()


def reference_key(reference: SourceRef) -> str:
    return stable_id(
        "ref",
        {
            "location": reference.location,
            "pages": reference.pages,
            "locator": reference.locator,
            "chunks": reference.chunk_ids,
            "character_range": reference.character_range,
        },
    )


_OFFICIAL_HOSTS = (
    "nvidia.com",
    "google.com",
    "cloud.google.com",
    "microsoft.com",
    "aws.amazon.com",
    "cxlconsortium.org",
    "tavily.com",
    "langchain.com",
)
_NEWS_HOSTS = ("reuters.com", "techcrunch.com", "bbc.com", "apnews.com")
_REPORT_HOSTS = ("gartner.com", "statista.com", "fortunebusinessinsights.com")


def classify_source(location: str) -> str:
    parsed = urlparse(location)
    if not parsed.scheme:
        return "paper" if location.lower().endswith(".pdf") else "unknown"
    host = (parsed.hostname or "").lower()

    def belongs(roots):
        return any(host == root or host.endswith("." + root) for root in roots)

    if belongs(("arxiv.org", "doi.org", "acm.org", "ieee.org")):
        return "paper"
    if host.endswith(".gov") or belongs(_OFFICIAL_HOSTS):
        return "official"
    if belongs(_NEWS_HOSTS):
        return "news"
    if belongs(_REPORT_HOSTS):
        return "report"
    if belongs(("medium.com",)) or "blog" in host.split("."):
        return "blog"
    return "unknown"


class SourceLoader:
    def __init__(
        self, *, root: Path = ROOT_DIR, extractor: TavilyExtractor | None = None
    ):
        self.root = Path(root).resolve()
        self.extractor = extractor
        self._local_cache: dict[tuple[str, str, tuple[int, ...]], SourceSnapshot] = {}
        self._lock = RLock()

    @traced_tool("source.fetch")
    def load(self, reference: SourceRef) -> SourceSnapshot:
        parsed = urlparse(reference.location)
        if parsed.scheme:
            snapshot = (
                (self.extractor or get_extractor())
                .extract([reference.location])
                .snapshots[0]
            )
            if reference.character_range is not None and snapshot.status == "ok":
                if reference.version != snapshot.content_hash:
                    raise ValueError(
                        "Source changed since the character span was selected"
                    )
                start, end = reference.character_range
                if end > len(snapshot.content):
                    raise ValueError("Character span exceeds the source body")
                return SourceSnapshot(
                    reference=reference,
                    acquisition=snapshot.acquisition,
                    status="ok",
                    content=snapshot.content[start:end],
                    retrieved_at=snapshot.retrieved_at,
                    provider_request_id=snapshot.provider_request_id,
                )
            return snapshot.model_copy(update={"reference": reference})
        path = Path(reference.location)
        path = (path if path.is_absolute() else self.root / path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("Local paper lies outside the configured source root")
        with self._lock:
            digest = fingerprint(path.read_bytes().hex())
            if reference.version and reference.version != digest:
                raise ValueError("PDF document version changed")
            actual_reference = reference.model_copy(update={"version": digest})
            pages = reference.pages or parse_pages(reference.locator)
            key = (str(path), digest, pages)
            if key in self._local_cache:
                emit(
                    "source_cache_hit",
                    details={"source_id": reference.source_id, "pages": pages},
                )
                return self._local_cache[key].model_copy(
                    update={"reference": actual_reference}
                )
            content = read_pdf_pages(path, pages=pages)
            snapshot = SourceSnapshot(
                reference=actual_reference,
                acquisition="local_pdf",
                content=content,
                status="ok" if content else "empty",
            )
            self._local_cache[key] = snapshot
            emit(
                "source_loaded",
                details={
                    "source_id": reference.source_id,
                    "pages": pages,
                    "characters": len(content),
                    "version": digest,
                },
            )
            return snapshot
