from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def make_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
