"""BGE-M3 임베딩 모델을 로드하는 모듈."""

from threading import RLock

from langchain_huggingface import HuggingFaceEmbeddings

from kv_cache_agent.config import EMBEDDING_MODEL, HF_TOKEN

_LOCK = RLock()
_EMBEDDER = None
_KEY = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """논문 RAG 파이프라인에서 공유할 BGE-M3 임베더를 반환한다."""
    global _EMBEDDER, _KEY
    with _LOCK:
        key = (EMBEDDING_MODEL, HF_TOKEN)
        if _EMBEDDER is None or _KEY != key:
            model_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
            _EMBEDDER = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            _KEY = key
        return _EMBEDDER


def clear_embedding_cache() -> None:
    global _EMBEDDER, _KEY
    with _LOCK:
        _EMBEDDER = _KEY = None
