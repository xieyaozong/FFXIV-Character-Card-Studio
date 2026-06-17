from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.answer_generator import answer_question


def main() -> None:
    question = " ".join(sys.argv[1:]) or "How should I prepare for raid mitigation?"
    result = answer_question(question)
    print(result["answer"])
    for citation in result["citations"]:
        print(f"- {citation['source']} score={citation['score']}")


if __name__ == "__main__":
    main()
