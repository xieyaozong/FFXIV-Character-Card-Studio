from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    source: str
    text: str


def load_markdown_docs(docs_dir: Path) -> list[Document]:
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        documents.append(Document(source=str(path), text=path.read_text(encoding="utf-8")))
    return documents
