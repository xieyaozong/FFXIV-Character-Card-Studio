from __future__ import annotations

from dataclasses import dataclass
from rag.document_loader import Document


@dataclass
class Chunk:
    chunk_id: str
    source: str
    text: str


def chunk_documents(documents: list[Document], chunk_size: int = 700) -> list[Chunk]:
    chunks = []
    for document in documents:
        paragraphs = [part.strip() for part in document.text.split("\n\n") if part.strip()]
        current = ""
        index = 0
        for paragraph in paragraphs:
            if len(current) + len(paragraph) > chunk_size and current:
                chunks.append(Chunk(f"{document.source}#{index}", document.source, current.strip()))
                current = ""
                index += 1
            current += paragraph + "\n\n"
        if current.strip():
            chunks.append(Chunk(f"{document.source}#{index}", document.source, current.strip()))
    return chunks
