"""FAISS에 저장된 논문 chunk를 검색하는 도구."""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from langchain_community.vectorstores import FAISS

from kv_cache_agent.config import PAPERS_DIR, VECTOR_DB_DIR
from kv_cache_agent.rag.vector_store import load_vector_store, search_vector_store

_store_lock = Lock()


@lru_cache(maxsize=2)
def _shared_store(path: Path):
    return load_vector_store(path)


def retrieve_paper_chunks(
    queries: list[str],
    top_k: int = 5,
    vector_store: FAISS | None = None,
    vector_db_path: Path = VECTOR_DB_DIR,
) -> list[dict[str, Any]]:
    """여러 질의로 논문 chunk를 검색하고 중복 결과를 제거한다."""
    # Initialize the BGE-M3 model/index once, including simultaneous first calls.
    with _store_lock:
        store = vector_store or _shared_store(vector_db_path)
    retrieved: list[dict[str, Any]] = []
    seen_chunk_ids: set[str] = set()

    for query in queries:
        for rank, (document, score) in enumerate(
            search_vector_store(store, query, top_k=top_k),
            start=1,
        ):
            metadata = dict(document.metadata)
            # A shared index may still contain its author's absolute PDF path.
            # Resolve only the project's known papers; leave the store untouched.
            indexed_path = Path(str(metadata.get("source_path", "")))
            paper_id = metadata.get("paper_id")
            if (
                paper_id in {"turboquant", "cxl_based_kv_cache"}
                and indexed_path.name == f"{paper_id}.pdf"
                and not indexed_path.is_file()
                and (local_paper := PAPERS_DIR / indexed_path.name).is_file()
            ):
                metadata["source_path"] = str(local_paper.resolve())
                if metadata.get("source_url") == str(indexed_path):
                    metadata["source_url"] = metadata["source_path"]
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
