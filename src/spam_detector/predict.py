"""Model loading and prediction helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import joblib
from sklearn.pipeline import Pipeline


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained model file is missing."""


Label = Literal["spam", "ham"]


@dataclass(frozen=True)
class Prediction:
    """Prediction result with label and confidence."""
    label: Label
    confidence: float  # 0..1


def load_model(path: Path) -> Pipeline:
    """Load a trained sklearn Pipeline from disk."""
    if not path.exists():
        raise ModelNotFoundError(
            f"Model file not found: {path}. "
            "Train first with: python -m spam_detector.train"
        )
    model = joblib.load(path)
    if not hasattr(model, "predict") or not hasattr(model, "predict_proba"):
        raise TypeError("Loaded object is not a compatible sklearn model"
                        "(missing predict/predict_proba).")
    return model


def predict_text(model: Pipeline, text: str) -> Prediction:
    """Predict spam/ham for a single message and return label + confidence."""
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("Input text is empty.")

    label: Label = model.predict([cleaned])[0]
    proba = model.predict_proba([cleaned])[0]
    confidence = float(max(proba))
    return Prediction(label=label, confidence=confidence)
