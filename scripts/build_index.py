"""Prepare the local PDF corpus once, outside the agent execution graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.tools.retriever import DEFAULT_EMBEDDING_MODEL, RetrievalError, build_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a BGE-M3 / FAISS PDF index")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--index-path", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--chunk-size", type=int, default=1400)
    parser.add_argument("--chunk-overlap", type=int, default=200)
    parser.add_argument("--force", action="store_true", help="Replace an existing prepared index")
    args = parser.parse_args(argv)
    try:
        manifest = build_index(
            args.raw_dir,
            args.index_path,
            embedding_model=args.embedding_model,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            force=args.force,
        )
    except (RetrievalError, ValueError, FileExistsError) as exc:
        parser.exit(2, f"Index preparation failed: {exc}\n")
    print(
        json.dumps({"index_path": str(args.index_path.resolve()), **manifest}, ensure_ascii=False, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
