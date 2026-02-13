from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from spam_detector.models import ModelName, TfidfConfig


@dataclass(frozen=True)
class AppConfig:
    model: ModelName = "nb"
    random_state: int = 42
    test_size: float = 0.2
    tfidf: TfidfConfig = TfidfConfig()


def load_config(path: Path) -> AppConfig:
    """Load configuration from a YAML file."""

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    tfidf_dict = data.get("tfidf") or {}
    tfidf = TfidfConfig(
        max_features=int(tfidf_dict.get("max_features", 5000)),
        ngram_range=tuple(tfidf_dict.get("ngram_range", [1, 2])),
    )

    return AppConfig(
        model=data.get("model", "nb"),
        random_state=int(data.get("random_state", 42)),
        test_size=float(data.get("test_size", 0.2)),
        tfidf=tfidf,
    )
