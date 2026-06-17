from __future__ import annotations


def format_citations(results: list[dict]) -> list[dict]:
    return [
        {
            "source": item["source"],
            "chunk_id": item["chunk_id"],
            "score": round(float(item["score"]), 4),
        }
        for item in results
    ]
