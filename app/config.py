from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from pathlib import Path


@dataclass
class Settings:
    docs_dir: Path = Path(getenv("RAG_DOCS_DIR", "docs_source"))
    vector_db_path: Path = Path(getenv("VECTOR_DB_PATH", "vector_db/index.pkl"))
    min_relevance: float = float(getenv("MIN_RELEVANCE", "0.12"))
    top_k: int = int(getenv("TOP_K", "4"))


settings = Settings()
