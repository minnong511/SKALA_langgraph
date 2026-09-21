"""BGE-M3 embeddings and a persistent FAISS index with JSON provenance.

The index is prepared separately by ``python -m scripts.build_index``. Loading a
retriever never re-embeds the corpus. There is no pickle deserialization.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.schemas import SourceDocument
from src.tools.web_search import stable_source_id

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-m3"
INDEX_VERSION = 1


class RetrievalError(RuntimeError):
    """The local corpus or its prepared index cannot be used."""


def _dependencies() -> tuple[Any, Any]:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise RetrievalError("Install the RAG dependencies: pip install -e '.[rag]'") from exc
    return faiss, np


def load_embedder(model_name: str) -> Any:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RetrievalError("sentence-transformers is required; install the rag extra") from exc
    return SentenceTransformer(model_name, trust_remote_code=False)


def _embed(embedder: Any, texts: list[str], np: Any) -> Any:
    vectors = np.asarray(
        embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
        dtype="float32",
    )
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise RetrievalError("Embedding model returned an invalid matrix shape")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.isfinite(vectors).all() or (norms == 0).any():
        raise RetrievalError("Embedding model returned non-finite or zero vectors")
    return np.ascontiguousarray(vectors / norms, dtype="float32")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_text(text: str, chunk_size: int = 1400, chunk_overlap: int = 200) -> Iterator[tuple[int, int, str]]:
    """Split within a page, preserving character offsets into the extracted text."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + chunk_size // 2, end)
            if boundary > start:
                end = boundary
        content = text[start:end]
        if content.strip():
            yield start, end, content
        if end >= len(text):
            break
        start = max(start + 1, end - chunk_overlap)


def collect_pdf_documents(
    raw_dir: str | Path, *, chunk_size: int = 1400, chunk_overlap: int = 200
) -> tuple[list[SourceDocument], dict[str, str]]:
    """Extract text pages; PDF creation timestamps are not publication dates."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RetrievalError("pypdf is required; install the rag extra") from exc
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    root = Path(raw_dir).resolve()
    if not root.is_dir():
        raise RetrievalError(f"PDF directory does not exist: {root}")
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf")
    if not files:
        raise RetrievalError(f"No PDF files found in {root}")
    documents: list[SourceDocument] = []
    fingerprints: dict[str, str] = {}
    seen: set[str] = set()
    now = datetime.now(UTC).isoformat()
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise RetrievalError(f"PDF resolves outside the raw directory: {path.name}")
        relative = path.relative_to(root).as_posix()
        file_hash = sha256_file(path)
        fingerprints[relative] = file_hash
        try:
            reader = PdfReader(path)
            title = str((reader.metadata or {}).get("/Title") or path.stem)
            for page_number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                for start, end, content in split_text(text, chunk_size, chunk_overlap):
                    source_id = stable_source_id("PDF", f"{file_hash}:{page_number}:{start}:{end}")
                    if source_id in seen:
                        continue
                    seen.add(source_id)
                    documents.append(
                        SourceDocument(
                            source_id=source_id,
                            source_type="paper",
                            title=title,
                            url="",
                            file_path=relative,
                            author=str((reader.metadata or {}).get("/Author") or ""),
                            content=content,
                            published_at="",
                            accessed_at=now,
                            page_or_section=f"p.{page_number}",
                            query="",
                            retrieval_method="FAISS",
                            metadata={
                                "local_path": relative,
                                "page": page_number,
                                "chunk_start": start,
                                "chunk_end": end,
                                "file_sha256": file_hash,
                                "publication_date_missing_reason": "PDF 메타데이터만으로 발행일을 확인할 수 없음",
                            },
                        )
                    )
        except Exception as exc:
            if isinstance(exc, RetrievalError):
                raise
            raise RetrievalError(f"Could not extract PDF: {relative} ({type(exc).__name__})") from exc
    if not documents:
        raise RetrievalError("The PDFs contain no extractable text; OCR scanned documents before indexing")
    return documents, fingerprints


def build_index(
    raw_dir: str | Path,
    index_path: str | Path,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    embedder: Any = None,
    chunk_size: int = 1400,
    chunk_overlap: int = 200,
    force: bool = False,
) -> dict[str, Any]:
    """Build an index once; refuse accidental replacement unless force is explicit."""
    target = Path(index_path)
    if (target / "manifest.json").exists() and not force:
        raise FileExistsError(f"An index already exists at {target}; use --force to rebuild")
    faiss, np = _dependencies()
    documents, fingerprints = collect_pdf_documents(
        raw_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    model = embedder if embedder is not None else load_embedder(embedding_model)
    vectors = _embed(model, [doc.content for doc in documents], np)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": INDEX_VERSION,
        "embedding_model": embedding_model,
        "dimension": int(vectors.shape[1]),
        "document_count": len(documents),
        "metric": "cosine",
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "created_at": datetime.now(UTC).isoformat(),
        "files": fingerprints,
    }
    # Publish manifest last. A interrupted rebuild is detected through file hashes.
    with tempfile.TemporaryDirectory(prefix=".build-", dir=target) as directory:
        temporary = Path(directory)
        faiss.write_index(index, str(temporary / "index.faiss"))
        (temporary / "documents.json").write_text(
            json.dumps([doc.model_dump(mode="json") for doc in documents], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        manifest["index_sha256"] = sha256_file(temporary / "index.faiss")
        manifest["documents_sha256"] = sha256_file(temporary / "documents.json")
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        for name in ("index.faiss", "documents.json", "manifest.json"):
            os.replace(temporary / name, target / name)
    return manifest


class FaissRetriever:
    def __init__(
        self, index_path: str | Path, *, embedding_model: str = DEFAULT_EMBEDDING_MODEL, embedder: Any = None
    ) -> None:
        faiss, self._np = _dependencies()
        root = Path(index_path)
        if not (root / "manifest.json").is_file():
            raise RetrievalError(f"No prepared index at {root}; run python -m scripts.build_index first")
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("version") != INDEX_VERSION:
                raise RetrievalError("Unsupported index version; rebuild the index")
            if manifest.get("embedding_model") != embedding_model:
                raise RetrievalError("Embedding model differs from the model used to build this index")
            for filename, key in (("index.faiss", "index_sha256"), ("documents.json", "documents_sha256")):
                if sha256_file(root / filename) != manifest.get(key):
                    raise RetrievalError(f"Index integrity check failed for {filename}; rebuild the index")
            data = json.loads((root / "documents.json").read_text(encoding="utf-8"))
            self._documents = [SourceDocument.model_validate(item) for item in data]
            self._index = faiss.read_index(str(root / "index.faiss"))
            if (
                self._index.ntotal != len(self._documents)
                or len(self._documents) != manifest["document_count"]
            ):
                raise RetrievalError("FAISS and document metadata counts do not match")
            if self._index.d != manifest["dimension"] or not self._documents:
                raise RetrievalError("FAISS dimension or document metadata is invalid")
        except RetrievalError:
            raise
        except Exception as exc:
            raise RetrievalError(f"Could not load prepared index ({type(exc).__name__})") from exc
        self.embedding_model = embedding_model
        self._embedder = embedder

    def search(self, query: str, limit: int = 5) -> list[SourceDocument]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Retrieval query must not be empty")
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("Retrieval limit must be a positive integer")
        if self._embedder is None:
            self._embedder = load_embedder(self.embedding_model)
        vector = _embed(self._embedder, [query.strip()], self._np)
        if vector.shape[1] != self._index.d:
            raise RetrievalError("Query embedding dimension does not match the prepared index")
        scores, indices = self._index.search(vector, min(limit, len(self._documents)))
        now = datetime.now(UTC).isoformat()
        result: list[SourceDocument] = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0:
                continue
            document = self._documents[int(index)]
            result.append(
                document.model_copy(
                    deep=True,
                    update={
                        "query": query.strip(),
                        "accessed_at": now,
                        "score": float(score),
                        "metadata": {**document.metadata, "score": float(score)},
                    },
                )
            )
        return result


Retriever = FaissRetriever
