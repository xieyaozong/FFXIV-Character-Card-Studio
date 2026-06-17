from __future__ import annotations

from app.config import settings
from rag.citation_formatter import format_citations
from rag.reranker import rerank
from rag.retriever import retrieve


def answer_question(question: str) -> dict:
    results = rerank(retrieve(question))
    relevant = [item for item in results if item["score"] >= settings.min_relevance]
    if not relevant:
        return {
            "answer": "I do not have enough source material in the local notes to answer that reliably.",
            "citations": [],
            "refused": True,
        }

    bullet_points = []
    for item in relevant[:2]:
        first_sentence = item["text"].replace("\n", " ").split(".")[0].strip()
        bullet_points.append(f"- {first_sentence}.")
    return {
        "answer": "Based on the local notes:\n" + "\n".join(bullet_points),
        "citations": format_citations(relevant),
        "refused": False,
    }
