from __future__ import annotations

from dataclasses import asdict
import numpy as np
from app.config import settings
from rag.vector_store import load_index


def retrieve(query: str, top_k: int | None = None) -> list[dict]:
    index = load_index(settings.vector_db_path)
    vector = index["vectorizer"].transform([query])
    scores = (index["matrix"] @ vector.T).toarray().ravel()
    order = np.argsort(scores)[::-1][: top_k or settings.top_k]
    results = []
    for position in order:
        score = float(scores[position])
        chunk = index["chunks"][position]
        results.append({**asdict(chunk), "score": score})
    return results
