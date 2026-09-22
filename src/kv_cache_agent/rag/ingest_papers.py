"""논문 PDF를 읽어 FAISS 인덱스로 저장하는 실행 모듈."""

from kv_cache_agent.config import PAPERS_DIR, VECTOR_DB_DIR
from kv_cache_agent.rag.chunker import load_and_split_papers
from kv_cache_agent.rag.vector_store import build_vector_store, save_vector_store


def ingest_papers() -> int:
    """논문을 임베딩하고 FAISS 인덱스를 저장한 뒤 chunk 수를 반환한다."""
    papers = sorted(PAPERS_DIR.glob("*.pdf"))
    if not papers:
        raise FileNotFoundError(f"논문 PDF를 찾을 수 없습니다: {PAPERS_DIR}")

    documents = load_and_split_papers(papers)
    vector_store = build_vector_store(documents)
    save_vector_store(vector_store, VECTOR_DB_DIR)

    print(f"논문 {len(papers)}건, chunk {len(documents)}개를 저장했습니다.")
    print(f"FAISS 경로: {VECTOR_DB_DIR}")
    return len(documents)


if __name__ == "__main__":
    ingest_papers()
