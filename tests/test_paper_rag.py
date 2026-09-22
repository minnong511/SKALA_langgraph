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
