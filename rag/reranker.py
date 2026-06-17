from __future__ import annotations


def rerank(results: list[dict]) -> list[dict]:
    return sorted(results, key=lambda item: item["score"], reverse=True)
