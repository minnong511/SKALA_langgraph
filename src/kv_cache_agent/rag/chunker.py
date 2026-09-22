"""논문 PDF를 페이지 정보와 함께 검색 단위로 나누는 모듈."""

from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

# BGE-M3에 넣기 적당한 크기가 되도록 문자 기준으로 분할한다.
DEFAULT_CHUNK_SIZE = 3_500
DEFAULT_CHUNK_OVERLAP = 500


def load_pdf_documents(pdf_path: Path) -> list[Document]:
    """PDF를 페이지 단위 Document 목록으로 읽는다."""
    reader = PdfReader(str(pdf_path))
    pdf_metadata = reader.metadata
    metadata_title = pdf_metadata.title if pdf_metadata else None
    pdf_title = str(metadata_title or pdf_path.stem)
    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue

        documents.append(
            Document(
                page_content=page_text,
                metadata={
                    "paper_id": pdf_path.stem,
                    "source_title": pdf_title,
                    "source_path": str(pdf_path),
                    "source_type": "paper",
                    "page": page_number,
                },
            )
        )

    return documents


def split_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """페이지 문서를 겹치는 chunk로 나누고 검색용 메타데이터를 추가한다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    for chunk_number, chunk in enumerate(chunks):
        paper_id = chunk.metadata.get("paper_id", "unknown")
        page_number = chunk.metadata.get("page", "unknown")
        chunk.metadata["chunk_id"] = f"{paper_id}:p{page_number}:c{chunk_number}"
        chunk.metadata["source_locator"] = f"p. {page_number}"

    return chunks


def load_and_split_papers(paper_paths: list[Path]) -> list[Document]:
    """여러 논문을 읽고 하나의 검색용 chunk 목록으로 합친다."""
    page_documents: list[Document] = []
    for paper_path in paper_paths:
        page_documents.extend(load_pdf_documents(paper_path))

    return split_documents(page_documents)
