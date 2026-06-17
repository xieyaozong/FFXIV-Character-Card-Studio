from __future__ import annotations

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.answer_generator import answer_question


def main() -> None:
    questions = json.loads(Path("eval/test_questions.json").read_text(encoding="utf-8"))
    for item in questions:
        result = answer_question(item["question"])
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
