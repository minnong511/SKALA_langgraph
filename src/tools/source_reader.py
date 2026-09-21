"""Reopen original PDFs and public web pages for independent evidence checks."""

from __future__ import annotations

import io
import ipaddress
import math
import re
import socket
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from src.schemas import SourceDocument
from src.tools.retriever import sha256_file
from src.tools.web_search import canonical_url


class SourceReadError(RuntimeError):
    """The original source cannot be safely read or independently confirmed."""


class _VisibleHTML(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._hidden = 0
        self.published_at = ""

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden += 1
        if tag == "meta":
            values = dict(attrs)
            name = (values.get("property") or values.get("name") or values.get("itemprop") or "").lower()
            if name in {
                "article:published_time",
                "datepublished",
                "citation_publication_date",
                "dc.date.issued",
            }:
                self.published_at = values.get("content") or self.published_at
        if tag in {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "tr", "section", "article"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "template"}:
            self._hidden = max(0, self._hidden - 1)
        if tag in {"p", "div", "li", "h1", "h2", "h3", "h4", "tr", "section", "article"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._hidden:
            self.parts.append(data)

    def text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def _public_url(url: str) -> str:
    try:
        normalized = canonical_url(url)
        parsed = urlsplit(normalized)
        host = parsed.hostname or ""
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise SourceReadError("Local or internal source hosts are not allowed")
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        if port not in {80, 443}:
            raise SourceReadError("Source URLs must use the standard HTTP or HTTPS port")
        addresses = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
        if not addresses:
            raise SourceReadError("Could not resolve source host")
        for entry in addresses:
            address = ipaddress.ip_address(entry[4][0].split("%", 1)[0])
            if not address.is_global:
                raise SourceReadError("Private, local and reserved source addresses are not allowed")
        # Canonicalization identifies equivalent sources, but original query order
        # can be meaningful (for example a publisher's signed download URL).
        original = urlsplit(url)
        return urlunsplit((original.scheme, original.netloc, original.path, original.query, ""))
    except (ValueError, OSError) as exc:
        raise SourceReadError("Source URL is invalid or its host cannot be resolved") from exc


class SourceReader:
    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        *,
        timeout_seconds: float = 20.0,
        client: Any = None,
        max_bytes: int = 10 * 1024 * 1024,
        max_content_chars: int = 200_000,
        max_redirects: int = 5,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be a positive finite number")
        if max_bytes < 1 or max_content_chars < 1 or max_redirects < 0:
            raise ValueError("Source read size limits must be positive and redirects nonnegative")
        self.raw_dir = Path(raw_dir).resolve()
        self.timeout_seconds = timeout_seconds
        self.max_bytes = max_bytes
        self.max_content_chars = max_content_chars
        self.max_redirects = max_redirects
        self._client = client

    def read(self, source: SourceDocument) -> SourceDocument:
        source = SourceDocument.model_validate(source)
        local_path = source.metadata.get("local_path") or source.file_path
        if local_path:
            text, details, publication_date = self._read_local(source, str(local_path))
        elif source.url:
            text, details, publication_date = self._read_web(source)
        else:
            raise SourceReadError("A source must have a URL or a local PDF path to reopen")
        if not text.strip():
            raise SourceReadError("The original source contains no extractable text")
        truncated = len(text) > self.max_content_chars
        metadata = {
            **source.metadata,
            **details,
            "original_read": True,
            "content_truncated": truncated,
            "original_content_length": len(text),
        }
        return source.model_copy(
            deep=True,
            update={
                "content": text[: self.max_content_chars],
                "metadata": metadata,
                "accessed_at": datetime.now(UTC).isoformat(),
                "published_at": publication_date or source.published_at,
            },
        )

    def _read_local(self, source: SourceDocument, value: str) -> tuple[str, dict[str, Any], str]:
        candidate = Path(value)
        path = (candidate if candidate.is_absolute() else self.raw_dir / candidate).resolve()
        if not path.is_relative_to(self.raw_dir):
            raise SourceReadError("Source path resolves outside the configured raw directory")
        if not path.is_file() or path.suffix.lower() != ".pdf":
            raise SourceReadError("Local original source must be an existing PDF")
        if path.stat().st_size > self.max_bytes:
            raise SourceReadError("Local source exceeds the configured size limit")
        expected_hash = source.metadata.get("file_sha256")
        if expected_hash and sha256_file(path) != expected_hash:
            raise SourceReadError("PDF has changed since indexing; rebuild the index before verification")
        text, page = self._pdf_text(path, source)
        return text, {"local_path": path.relative_to(self.raw_dir).as_posix(), "original_page": page}, ""

    @staticmethod
    def _pdf_text(data: Any, source: SourceDocument) -> tuple[str, int | None]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise SourceReadError("pypdf is required to reopen PDF sources; install the rag extra") from exc
        page = source.metadata.get("page")
        if page is None:
            match = re.fullmatch(r"p\.?\s*(\d+)", source.page_or_section.strip())
            if match:
                page = int(match.group(1))
        try:
            reader = PdfReader(data)
            if page is not None:
                page = int(page)
                if not 1 <= page <= len(reader.pages):
                    raise SourceReadError("The cited PDF page does not exist")
                return reader.pages[page - 1].extract_text() or "", page
            return "\n\n".join(
                f"[p.{number}]\n{item.extract_text() or ''}" for number, item in enumerate(reader.pages, 1)
            ), None
        except SourceReadError:
            raise
        except Exception as exc:
            raise SourceReadError(f"Could not read original PDF ({type(exc).__name__})") from exc

    def _read_web(self, source: SourceDocument) -> tuple[str, dict[str, Any], str]:
        client = self._client or httpx.Client(follow_redirects=False, trust_env=False)
        url = source.url
        try:
            for redirects in range(self.max_redirects + 1):
                url = _public_url(url)
                with client.stream(
                    "GET",
                    url,
                    follow_redirects=False,
                    timeout=self.timeout_seconds,
                    headers={
                        "User-Agent": "SKALA-Research/1.0",
                        "Accept": "text/html,text/plain,application/pdf",
                    },
                ) as response:
                    if response.status_code in {301, 302, 303, 307, 308}:
                        location = response.headers.get("location")
                        if not location or redirects >= self.max_redirects:
                            raise SourceReadError("Source redirect limit exceeded or redirect target missing")
                        url = urljoin(url, location)
                        continue
                    response.raise_for_status()
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    body = bytearray()
                    for part in response.iter_bytes():
                        body.extend(part)
                        if len(body) > self.max_bytes:
                            raise SourceReadError("Remote source exceeds the configured size limit")
                    raw = bytes(body)
                    details: dict[str, Any] = {"original_url": url, "content_type": content_type}
                    if content_type == "application/pdf" or raw.startswith(b"%PDF-"):
                        text, page = self._pdf_text(io.BytesIO(raw), source)
                        return text, {**details, "original_page": page}, ""
                    if content_type and content_type not in {
                        "text/html",
                        "application/xhtml+xml",
                        "text/plain",
                        "text/markdown",
                    }:
                        raise SourceReadError(f"Unsupported original source content type: {content_type}")
                    decoded = raw.decode(response.encoding or "utf-8", errors="replace")
                    if content_type in {"text/plain", "text/markdown"}:
                        return decoded, details, ""
                    parser = _VisibleHTML()
                    parser.feed(decoded)
                    return parser.text(), details, parser.published_at
            raise SourceReadError("Could not follow source redirects")
        except httpx.HTTPError as exc:
            raise SourceReadError(f"Could not fetch original source ({type(exc).__name__})") from exc
        finally:
            if self._client is None:
                client.close()
