from __future__ import annotations

from pathlib import Path
from app.config import settings
from rag.answer_generator import answer_question
from rag.chunker import chunk_documents
from rag.document_loader import load_markdown_docs
from rag.vector_store import build_index


def test_rag_answers_with_citation(tmp_path, monkeypatch) -> None:
    docs = load_markdown_docs(Path("docs_source"))
    chunks = chunk_documents(docs)
    index_path = tmp_path / "index.pkl"
    build_index(chunks, index_path)
    monkeypatch.setattr(settings, "vector_db_path", index_path)
    result = answer_question("What should I check before crafting?")
    assert result["refused"] is False
    assert result["citations"]


def test_rag_refuses_unknown_question(tmp_path, monkeypatch) -> None:
    docs = load_markdown_docs(Path("docs_source"))
    chunks = chunk_documents(docs)
    index_path = tmp_path / "index.pkl"
    build_index(chunks, index_path)
    monkeypatch.setattr(settings, "vector_db_path", index_path)
    result = answer_question("Explain submarine engine maintenance.")
    assert result["refused"] is True
