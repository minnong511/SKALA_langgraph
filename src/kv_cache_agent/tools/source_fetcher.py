"""웹 페이지와 PDF 원문을 수집하고 공통 형식으로 반환하는 도구."""

import json
from datetime import UTC, datetime
from io import BytesIO
from pathlib import PurePosixPath
from typing import Literal
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from kv_cache_agent.schemas.tool_outputs import FetchedSource

DEFAULT_TIMEOUT = 15.0
DEFAULT_MAX_CHARS = 30_000
USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153.0.0.0 Safari/537.36"
# USER_AGENT = "kv-cache-agent/0.1 source-fetcher"
BLOCKED_STATUS_CODES = {401, 403, 451}


def _error_result(
    url: str,
    message: str,
    *,
    status_code: int = 0,
    content_type: str = "",
    fetch_status: Literal["error", "blocked"] = "error",
) -> FetchedSource:
    """원문 수집 실패도 에이전트가 처리할 수 있는 결과로 표현한다."""
    return {
        "title": "",
        "url": url,
        "content": "",
        "source_type": "web",
        "published_date": "",
        "content_length": 0,
        "status_code": status_code,
        "content_type": content_type,
        "fetch_status": fetch_status,
        "error": message,
    }


def _extract_html(response: httpx.Response, url: str) -> tuple[str, str]:
    """HTML에서 제목과 본문 텍스트를 추출한다."""
    soup = BeautifulSoup(response.text, "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()

    title_tag = soup.find("title")
    title = title_tag.get_text(" ", strip=True) if title_tag else url
    content_root = soup.find("main") or soup.find("article") or soup.body or soup
    content = content_root.get_text(" ", strip=True)
    return title, content


def _html_metadata(response: httpx.Response) -> dict[str, str]:
    """실제로 페이지에 표시된 날짜·작성자·기관만 추출한다.

    JSON-LD는 본문 정리 과정에서 제거되므로 먼저 읽는다. 게시 날짜가 없으면
    빈 값으로 유지하며 서버 응답의 Date를 게시일로 오인하지 않는다.
    """
    soup = BeautifulSoup(response.text, "html.parser")
    metadata: dict[str, str] = {}
    names = {
        "article:published_time": "published_date",
        "datepublished": "published_date",
        "citation_publication_date": "published_date",
        "date": "published_date",
        "author": "authors",
        "citation_author": "authors",
        "og:site_name": "publisher",
    }
    for tag in soup.find_all("meta"):
        name = str(tag.get("property") or tag.get("name") or "").lower()
        value = str(tag.get("content") or "").strip()
        if name in names and value:
            metadata.setdefault(names[name], value)
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = json.loads(tag.get_text())
        except (ValueError, TypeError):
            continue
        records = data if isinstance(data, list) else [data]
        for item in records:
            if not isinstance(item, dict):
                continue
            for node in [item, *item.get("@graph", [])]:
                if not isinstance(node, dict):
                    continue
                if node.get("datePublished"):
                    metadata.setdefault("published_date", str(node["datePublished"]))
                for key, field in (("author", "authors"), ("publisher", "publisher")):
                    value = node.get(key)
                    people = value if isinstance(value, list) else [value]
                    label = ", ".join(
                        str(p.get("name", "")) if isinstance(p, dict) else p
                        for p in people
                        if isinstance(p, (str, dict))
                    )
                    if label:
                        metadata.setdefault(field, label)
    return metadata


def _extract_pdf(response: httpx.Response, url: str) -> tuple[str, str]:
    """PDF의 메타데이터 제목과 페이지별 텍스트를 추출한다."""
    reader = PdfReader(BytesIO(response.content))
    metadata = reader.metadata
    title = str(metadata.title or "") if metadata else ""
    if not title:
        title = PurePosixPath(urlparse(url).path).stem or url

    page_texts = [(page.extract_text() or "").strip() for page in reader.pages]
    return title, "\n\n".join(text for text in page_texts if text)


def _is_pdf(content_type: str, url: str) -> bool:
    """응답 헤더 또는 URL 확장자로 PDF 여부를 판단한다."""
    path = urlparse(url).path.lower()
    return "application/pdf" in content_type.lower() or path.endswith(".pdf")


def fetch_source(
    url: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.Client | None = None,
) -> FetchedSource:
    """웹 페이지 또는 PDF를 가져와 에이전트용 원문 형식으로 반환한다.

    허용 범위는 HTTP/HTTPS URL이며, 원문이 지나치게 길어지는 것을 막기
    위해 ``max_chars`` 이후의 텍스트는 잘라낸다. 테스트나 상위 계층에서
    HTTP 클라이언트를 주입할 수 있다.
    """
    clean_url = url.strip()
    parsed_url = urlparse(clean_url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        return _error_result(clean_url, "HTTP/HTTPS URL만 지원합니다.")
    if max_chars < 1:
        raise ValueError("max_chars는 1 이상이어야 합니다.")
    if timeout <= 0:
        raise ValueError("timeout은 0보다 커야 합니다.")

    owns_client = client is None
    http_client = client or httpx.Client(
        follow_redirects=True,
        timeout=timeout,
        headers={"User-Agent": USER_AGENT},
    )

    try:
        response = http_client.get(clean_url)
        content_type = response.headers.get("content-type", "")
        if response.status_code in BLOCKED_STATUS_CODES:
            return _error_result(
                clean_url,
                f"원문 접근이 차단되었습니다: HTTP {response.status_code}",
                status_code=response.status_code,
                content_type=content_type,
                fetch_status="blocked",
            )
        response.raise_for_status()

        metadata = {}
        if _is_pdf(content_type, clean_url):
            title, content = _extract_pdf(response, clean_url)
            source_type = "pdf"
        elif "text/html" in content_type.lower() or not content_type:
            metadata = _html_metadata(response)
            title, content = _extract_html(response, clean_url)
            source_type = "web"
        else:
            return {
                **_error_result(
                    clean_url,
                    f"지원하지 않는 콘텐츠 형식입니다: {content_type}",
                    status_code=response.status_code,
                    content_type=content_type,
                ),
                "fetch_status": "unsupported",
            }

        content = content[:max_chars]
        return {
            "title": title,
            "url": clean_url,
            "content": content,
            "source_type": source_type,
            "published_date": metadata.get("published_date", ""),
            "authors": metadata.get("authors", ""),
            "publisher": metadata.get("publisher", urlparse(clean_url).netloc),
            "retrieved_at": datetime.now(tz=UTC).date().isoformat(),
            "content_length": len(content),
            "status_code": response.status_code,
            "content_type": content_type,
            "fetch_status": "ok" if content else "empty",
        }
    except httpx.HTTPStatusError as error:
        error_response = error.response
        return _error_result(
            clean_url,
            f"원문 수집에 실패했습니다: {error}",
            status_code=error_response.status_code if error_response else 0,
            content_type=(
                error_response.headers.get("content-type", "") if error_response else ""
            ),
        )
    except (httpx.HTTPError, ValueError) as error:
        return _error_result(
            clean_url,
            f"원문 수집에 실패했습니다: {error}",
        )
    except Exception as error:  # noqa: BLE001
        return _error_result(
            clean_url,
            f"원문 파싱에 실패했습니다: {error}",
        )
    finally:
        if owns_client:
            http_client.close()
