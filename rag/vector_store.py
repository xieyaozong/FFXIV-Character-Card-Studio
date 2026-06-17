from __future__ import annotations

from pathlib import Path
import pickle
from rag.chunker import Chunk
from rag.embedder import make_vectorizer


def build_index(chunks: list[Chunk], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vectorizer = make_vectorizer()
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    with output_path.open("wb") as handle:
        pickle.dump({"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}, handle)


def load_index(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle)
