from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    path = Path("vector_db")
    for item in path.glob("*"):
        if item.name in {"README.md", ".gitkeep"}:
            continue
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()
    print("cleaned vector_db")


if __name__ == "__main__":
    main()
