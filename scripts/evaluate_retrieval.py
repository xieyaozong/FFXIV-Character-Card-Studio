from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.retriever import retrieve


def main() -> None:
    questions = json.loads(Path("eval/test_questions.json").read_text(encoding="utf-8"))
    for item in questions:
        results = retrieve(item["question"])
        print(item["question"], "->", [Path(result["source"]).name for result in results[:2]])


if __name__ == "__main__":
    main()
