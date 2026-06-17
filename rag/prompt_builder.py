from __future__ import annotations


def build_prompt(question: str, chunks: list[dict]) -> str:
    context = "\n\n".join(f"[{idx + 1}] {chunk['text']}" for idx, chunk in enumerate(chunks))
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations."
