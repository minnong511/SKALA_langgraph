from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from kv_cache_agent.agents.technical import _build_evidence_cards
from kv_cache_agent.rag.chunker import split_documents
from kv_cache_agent.rag.vector_store import (
    build_vector_store,
    load_vector_store,
    save_vector_store,
    search_vector_store,
)
from kv_cache_agent.schemas.technical import TechnicalExtraction, TechnicalFinding


def test_shared_index_pdf_path_resolves_to_current_checkout(monkeypatch, tmp_path):
    from reportlab.pdfgen import canvas

    from kv_cache_agent.agents.verifier import _fetch_original_source
    from kv_cache_agent.tools import paper_retriever

    paper_dir = tmp_path / "papers"
    paper_dir.mkdir()
    pdf = paper_dir / "turboquant.pdf"
    writer = canvas.Canvas(str(pdf))
    writer.drawString(50, 700, "TurboQuant compresses KV Cache.")
    writer.save()
    old_path = str(tmp_path / "old_checkout" / "turboquant.pdf")
    document = Document(
        page_content="TurboQuant compresses KV Cache.",
        metadata={
            "paper_id": "turboquant",
            "source_title": "TurboQuant",
            "source_path": old_path,
            "source_locator": "p. 1",
            "chunk_id": "turboquant:p1:c0",
        },
    )
    store = build_vector_store([document], FakeEmbeddings())
    monkeypatch.setattr(paper_retriever, "PAPERS_DIR", paper_dir)
    chunks = paper_retriever.retrieve_paper_chunks(
        ["TurboQuant"], top_k=1, vector_store=store
    )
    extraction = TechnicalExtraction(
        summary="Source path integration",
        limitations=[],
        findings=[
            TechnicalFinding(
                technology="TurboQuant",
                claim="TurboQuant compresses KV Cache.",
                evidence_text="TurboQuant compresses KV Cache.",
                source_chunk_ids=["turboquant:p1:c0"],
                claim_type="fact",
                confidence=0.9,
            )
        ],
    )
    cards, missing = _build_evidence_cards(extraction, chunks)
    assert not missing
    assert cards[0]["source_url"] == str(pdf)
    source = _fetch_original_source(cards[0])
    assert source["fetch_status"] == "ok"
    assert "TurboQuant compresses KV Cache" in source["content"]
    assert document.metadata["source_path"] == old_path


class FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        normalized = text.lower()
        if "turboquant" in normalized:
            return [1.0, 0.0]
        if "cxl" in normalized:
            return [0.0, 1.0]
        return [0.5, 0.5]


def test_split_documents_preserves_paper_metadata() -> None:
    documents = [
        Document(
            page_content="TurboQuant " * 100,
            metadata={
                "paper_id": "turboquant",
                "source_title": "TurboQuant",
                "page": 4,
            },
        )
    ]

    chunks = split_documents(documents, chunk_size=100, chunk_overlap=20)

    assert chunks
    assert chunks[0].metadata["paper_id"] == "turboquant"
    assert chunks[0].metadata["source_locator"] == "p. 4"
    assert chunks[0].metadata["chunk_id"].startswith("turboquant:p4:")


def test_faiss_store_can_search_save_and_load(tmp_path) -> None:
    documents = [
        Document(page_content="TurboQuant compresses KV Cache."),
        Document(page_content="CXL expands memory for KV Cache."),
    ]
    embeddings = FakeEmbeddings()

    vector_store = build_vector_store(documents, embeddings)
    hits = search_vector_store(vector_store, "TurboQuant", top_k=1)
    assert "TurboQuant" in hits[0][0].page_content

    save_vector_store(vector_store, tmp_path)
    loaded_store = load_vector_store(tmp_path, embeddings)
    loaded_hits = search_vector_store(loaded_store, "CXL", top_k=1)
    assert "CXL" in loaded_hits[0][0].page_content


def test_technical_finding_is_converted_to_evidence_card() -> None:
    extraction = TechnicalExtraction(
        summary="기술 요약",
        limitations=[],
        findings=[
            TechnicalFinding(
                technology="TurboQuant",
                claim="TurboQuant는 KV Cache를 압축한다.",
                evidence_text="논문 근거",
                source_chunk_ids=["turboquant:p4:c0"],
                claim_type="fact",
                confidence=0.9,
            )
        ],
    )
    retrieved_chunks = [
        {
            "chunk_id": "turboquant:p4:c0",
            "content": "논문 내용",
            "metadata": {
                "source_title": "TurboQuant",
                "source_path": "data/papers/turboquant.pdf",
                "source_locator": "p. 4",
            },
        }
    ]

    cards, missing_sources = _build_evidence_cards(
        extraction,
        retrieved_chunks,
    )

    assert not missing_sources
    assert cards[0]["retrieval_method"] == "faiss"
    assert cards[0]["verification_status"] == "unverified"


def test_parallel_retrieval_initializes_shared_store_once(monkeypatch, tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from unittest.mock import Mock

    from kv_cache_agent.tools import paper_retriever

    documents = [
        Document(page_content="TurboQuant evidence", metadata={"chunk_id": "turbo:p1"}),
        Document(page_content="CXL evidence", metadata={"chunk_id": "cxl:p1"}),
    ]
    store = build_vector_store(documents, FakeEmbeddings())
    loader = Mock(return_value=store)
    monkeypatch.setattr(paper_retriever, "load_vector_store", loader)
    paper_retriever._shared_store.cache_clear()
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(
                pool.map(
                    lambda query: paper_retriever.retrieve_paper_chunks(
                        [query],
                        top_k=1,
                        vector_db_path=tmp_path,
                    ),
                    ["TurboQuant", "CXL", "TurboQuant", "CXL"],
                )
            )
        assert loader.call_count == 1
        assert [items[0]["chunk_id"] for items in results] == [
            "turbo:p1",
            "cxl:p1",
            "turbo:p1",
            "cxl:p1",
        ]
    finally:
        paper_retriever._shared_store.cache_clear()
