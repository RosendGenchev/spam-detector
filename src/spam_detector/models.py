from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

ModelName = Literal["nb", "logreg"]


@dataclass(frozen=True)
class TfidfConfig:
    """TF-IDF vectorizer configuration."""

    max_features: int = 5000
    ngram_range: tuple[int, int] = (1, 2)


def build_pipeline(model: ModelName, tfidf: TfidfConfig | None = None) -> Pipeline:
    """Create a scikit-learn Pipeline for text -> spam/ham classification."""

    tfidf = tfidf or TfidfConfig()

    vectorizer = TfidfVectorizer(
        max_features=tfidf.max_features,
        ngram_range=tfidf.ngram_range,
        stop_words="english",
    )

    if model == "nb":
        clf = MultinomialNB()
    elif model == "logreg":
        clf = LogisticRegression(max_iter=2000)
    else:  # pragma: no cover
        raise ValueError(f"Unknown model name: {model}")

    return Pipeline([( "tfidf", vectorizer), ("clf", clf)])
