"""Tool contracts exercised without external requests or embedding downloads."""

from __future__ import annotations

import json
import socket
from itertools import pairwise

import httpx
import pytest

from src.schemas import SourceDocument
from src.tools.retriever import FaissRetriever, RetrievalError, build_index, collect_pdf_documents, split_text
from src.tools.source_reader import SourceReader, SourceReadError
from src.tools.web_search import SearchError, TavilySearch


def test_tavily_deduplicates_and_source_ids_do_not_depend_on_query_or_rank():
    requests = []
    items = [
        {
            "url": "https://example.org/paper?utm_source=search#result",
            "title": "Paper",
            "content": "excerpt",
            "raw_content": "original text",
            "published_date": "2026-01-02",
        },
        {"url": "https://example.org/paper", "content": "duplicate"},
        {"url": "https://example.org/other", "content": "another source"},
        {"url": "file:///etc/passwd", "content": "invalid"},
    ]

    def respond(request):
        requests.append(request)
        payload = json.loads(request.content)
        return httpx.Response(
            200, json={"results": items if payload["query"] == "first" else list(reversed(items))}
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        search = TavilySearch("test-only-secret", client=client)
        first = search.search("first")
        second = search.search("second")
    assert len(first) == len(second) == 2
    assert {source.source_id for source in first} == {source.source_id for source in second}
    assert first[0].content == "original text"
    assert first[0].published_at == "2026-01-02"
    assert first[0].accessed_at and first[0].retrieval_method == "Tavily"
    assert first[0].query == "first"
    assert json.loads(requests[0].content)["include_published_date"] is True
    assert requests[0].extensions["timeout"]["read"] == 20.0


def test_tavily_configuration_and_http_errors_are_explicit():
    with pytest.raises(ValueError, match="TAVILY_API_KEY"):
        TavilySearch("")
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as client:
        search = TavilySearch("secret-should-not-be-logged", client=client)
        with pytest.raises(ValueError, match="empty"):
            search.search(" ")
        with pytest.raises(SearchError) as error:
            search.search("query")
        assert "secret-should-not-be-logged" not in str(error.value)


@pytest.fixture
def public_dns(monkeypatch):
    def resolve(host, port, **kwargs):
        ip = host if host in {"127.0.0.1", "10.0.0.1"} else "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (ip, port))]

    monkeypatch.setattr("src.tools.source_reader.socket.getaddrinfo", resolve)


def test_reader_fetches_original_text_and_published_date(public_dns, tmp_path):
    body = '<html><head><meta property="article:published_time" content="2026-04-01" /></head><body><h1>Original paper</h1><p>Measured latency was 10 ms.</p><script>False evidence.</script></body></html>'
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, text=body, headers={"content-type": "text/html"})
        )
    ) as client:
        reader = SourceReader(tmp_path, client=client)
        initial = SourceDocument(source_id="WEB-a", url="https://example.org/paper", content="search summary")
        original = reader.read(initial)
    assert "Measured latency was 10 ms." in original.content
    assert "False evidence" not in original.content
    assert original.published_at == "2026-04-01"
    assert original.metadata["original_read"] is True
    assert initial.content == "search summary"


def test_reader_rejects_private_redirect_before_request(public_dns, tmp_path):
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://127.0.0.1/internal"})

    with (
        httpx.Client(transport=httpx.MockTransport(respond)) as client,
        pytest.raises(SourceReadError, match="Private"),
    ):
        SourceReader(tmp_path, client=client).read(
            SourceDocument(source_id="WEB-a", url="https://example.org/paper")
        )
    assert requests == ["https://example.org/paper"]


def test_reader_rejects_local_path_escape(tmp_path):
    with pytest.raises(SourceReadError, match="outside"):
        SourceReader(tmp_path / "raw").read(SourceDocument(source_id="PDF-a", file_path="../secret.pdf"))


def test_reader_enforces_response_size(public_dns, tmp_path):
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="X" * 100))
        ) as client,
        pytest.raises(SourceReadError, match="size limit"),
    ):
        SourceReader(tmp_path, client=client, max_bytes=20).read(
            SourceDocument(source_id="WEB-a", url="https://example.org/paper")
        )


def test_chunks_cover_document_with_offsets_and_bounded_overlap():
    text = "evidence sentence " * 80
    chunks = list(split_text(text, chunk_size=100, chunk_overlap=20))
    assert chunks[0][0] == 0 and chunks[-1][1] == len(text)
    assert all(content == text[start:end] and len(content) <= 100 for start, end, content in chunks)
    assert all(left[1] >= right[0] for left, right in pairwise(chunks))
    with pytest.raises(ValueError):
        list(split_text(text, 10, 10))


class FakeEmbedder:
    def encode(self, texts, **kwargs):
        import numpy as np

        return np.array(
            [
                [
                    1.0 if "quantization" in text.lower() else 0.0,
                    1.0 if "market" in text.lower() else 0.0,
                    0.1,
                ]
                for text in texts
            ],
            dtype="float32",
        )


@pytest.fixture
def pdf_corpus(tmp_path):
    from reportlab.pdfgen import canvas

    raw = tmp_path / "raw"
    raw.mkdir()
    file = raw / "research.pdf"
    pdf = canvas.Canvas(str(file))
    pdf.setTitle("Research evidence")
    pdf.setAuthor("Test Author")
    pdf.drawString(72, 750, "Quantization lowers memory usage under measured conditions.")
    pdf.showPage()
    pdf.drawString(72, 750, "Market adoption depends on price and service contracts.")
    pdf.save()
    return raw


def test_pdf_index_retrieval_and_independent_page_verification(pdf_corpus, tmp_path):
    pytest.importorskip("faiss")
    target = tmp_path / "index"
    embedder = FakeEmbedder()
    manifest = build_index(pdf_corpus, target, embedder=embedder)
    assert manifest["document_count"] == 2
    assert {p.name for p in target.iterdir()} == {"index.faiss", "documents.json", "manifest.json"}
    retriever = FaissRetriever(target, embedder=embedder)
    source = retriever.search("quantization", limit=1)[0]
    assert source.page_or_section == "p.1"
    assert source.file_path == "research.pdf"
    assert source.author == "Test Author"
    assert source.published_at == ""  # PDF creation time is not a publication date.
    assert source.score > 0.99
    assert source.source_id == retriever.search("quantization memory", limit=2)[0].source_id
    original = SourceReader(pdf_corpus).read(source)
    assert source.content in original.content
    assert "Market adoption" not in original.content
    assert original.metadata["original_page"] == 1
    with pytest.raises(FileExistsError):
        build_index(pdf_corpus, target, embedder=embedder)
    with pytest.raises(RetrievalError, match="Embedding model differs"):
        FaissRetriever(target, embedding_model="different-model", embedder=embedder)
    (target / "documents.json").write_text("[]", encoding="utf-8")
    with pytest.raises(RetrievalError, match="integrity"):
        FaissRetriever(target, embedder=embedder)


def test_pdf_content_change_prevents_stale_verification(pdf_corpus):
    sources, _ = collect_pdf_documents(pdf_corpus)
    with (pdf_corpus / "research.pdf").open("ab") as stream:
        stream.write(b"\n%changed")
    with pytest.raises(SourceReadError, match="changed since indexing"):
        SourceReader(pdf_corpus).read(sources[0])


def test_pdf_collection_skips_duplicate_copies_and_rejects_missing_corpus(pdf_corpus, tmp_path):
    (pdf_corpus / "copy.pdf").write_bytes((pdf_corpus / "research.pdf").read_bytes())
    documents, fingerprints = collect_pdf_documents(pdf_corpus)
    assert len(documents) == 2 and len(fingerprints) == 2
    with pytest.raises(RetrievalError, match="does not exist"):
        collect_pdf_documents(tmp_path / "missing")
