from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from rag.chunker import chunk_documents
from rag.document_loader import load_markdown_docs
from rag.vector_store import build_index


def main() -> None:
    documents = load_markdown_docs(settings.docs_dir)
    chunks = chunk_documents(documents)
    build_index(chunks, settings.vector_db_path)
    print(f"indexed {len(chunks)} chunks -> {settings.vector_db_path}")


if __name__ == "__main__":
    main()
