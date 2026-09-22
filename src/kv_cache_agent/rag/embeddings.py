"""BGE-M3 임베딩 모델을 로드하는 모듈."""

from langchain_huggingface import HuggingFaceEmbeddings

from kv_cache_agent.config import EMBEDDING_MODEL, HF_TOKEN


def get_embeddings() -> HuggingFaceEmbeddings:
    """논문 RAG 파이프라인에서 공유할 BGE-M3 임베더를 반환한다."""
    model_kwargs = {"token": HF_TOKEN} if HF_TOKEN else {}
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs=model_kwargs,
        # 문서와 질의 벡터를 정규화하여 유사도 검색을 안정화한다.
        encode_kwargs={"normalize_embeddings": True},
    )
