"""FAISS 인덱스 생성, 저장, 로드, 검색을 담당하는 모듈."""

from collections import OrderedDict
from pathlib import Path
from threading import RLock

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from kv_cache_agent.rag.embeddings import get_embeddings

_LOCK = RLock()
_STORES: OrderedDict[tuple, FAISS] = OrderedDict()


def clear_vector_store_cache(path: Path | None = None) -> None:
    with _LOCK:
        for key in list(_STORES):
            if path is None or key[0] == str(path.resolve()):
                _STORES.pop(key)


def build_vector_store(
    documents: list[Document],
    embeddings: Embeddings | None = None,
) -> FAISS:
    """논문 chunk 목록으로 메모리 내 FAISS 인덱스를 생성한다."""
    embedder = embeddings or get_embeddings()
    return FAISS.from_documents(documents, embedder)


def save_vector_store(vector_store: FAISS, path: Path) -> None:
    """FAISS 인덱스를 지정한 로컬 디렉터리에 저장한다."""
    path.mkdir(parents=True, exist_ok=True)
    with _LOCK:
        vector_store.save_local(str(path))
        clear_vector_store_cache(path)


def load_vector_store(
    path: Path,
    embeddings: Embeddings | None = None,
) -> FAISS:
    """프로젝트가 직접 생성한 로컬 FAISS 인덱스를 로드한다."""
    if not (path / "index.faiss").exists() or not (path / "index.pkl").exists():
        raise FileNotFoundError(
            f"FAISS 인덱스를 찾을 수 없습니다: {path}. 먼저 ingest_papers를 실행하세요."
        )

    embedder = embeddings or get_embeddings()
    with _LOCK:
        stamps = tuple(
            (file.stat().st_mtime_ns, file.stat().st_ctime_ns, file.stat().st_size)
            for file in (path / "index.faiss", path / "index.pkl")
        )
        key = (str(path.resolve()), stamps, id(embedder))
        if key not in _STORES:
            _STORES[key] = FAISS.load_local(
                str(path),
                embedder,
                # Only load locally generated project indexes.
                allow_dangerous_deserialization=True,
            )
        _STORES.move_to_end(key)
        while len(_STORES) > 4:
            _STORES.popitem(last=False)
        return _STORES[key]


def search_vector_store(
    vector_store: FAISS,
    query: str,
    top_k: int = 5,
) -> list[tuple[Document, float]]:
    """질의와 가까운 논문 chunk를 거리 점수와 함께 반환한다."""
    if top_k < 1:
        raise ValueError("top_k는 1 이상이어야 합니다.")

    return vector_store.similarity_search_with_score(query, k=top_k)
