"""FAISS에 저장된 논문 chunk를 검색하는 도구."""

from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from kv_cache_agent.config import VECTOR_DB_DIR
from kv_cache_agent.rag.vector_store import load_vector_store, search_vector_store


def retrieve_paper_chunks(
    queries: list[str],
    top_k: int = 5,
    vector_store: FAISS | None = None,
    vector_db_path: Path = VECTOR_DB_DIR,
) -> list[dict[str, Any]]:
    """여러 질의로 논문 chunk를 검색하고 중복 결과를 제거한다."""
    store = vector_store or load_vector_store(vector_db_path)
    retrieved: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for query in queries:
        for rank, (document, score) in enumerate(
            search_vector_store(store, query, top_k=top_k),
            start=1,
        ):
            metadata = dict(document.metadata)
            chunk_id = str(
                metadata.get(
                    "chunk_id",
                    f"{metadata.get('paper_id', 'unknown')}:"
                    f"p{metadata.get('page', 'unknown')}:r{rank}",
                )
            )
            if chunk_id in seen_chunk_ids:
                continue

            seen_chunk_ids.add(chunk_id)
            retrieved.append(
                {
                    "chunk_id": chunk_id,
                    "query": query,
                    "rank": rank,
                    "score": float(score),
                    "content": document.page_content,
                    "metadata": metadata,
                }
            )

    return retrieved
