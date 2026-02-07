from pathlib import Path

import pytest

from spam_detector.predict import load_model, predict_text, ModelNotFoundError


def test_load_model_missing_raises() -> None:
    with pytest.raises(ModelNotFoundError):
        load_model(Path("model/does_not_exist.joblib"))


def test_predict_text_empty_raises(tmp_path) -> None:
    # We don't need a real model for this test: it should fail before prediction.
    class Dummy:
        def predict(self, x):
            return ["ham"]

        def predict_proba(self, x):
            return [[0.9, 0.1]]

    with pytest.raises(ValueError):
        predict_text(Dummy(), "   ")
        
def test_predict_text_returns_label_and_confidence() -> None:
    class Dummy:
        def predict(self, x):
            return ["spam"]

        def predict_proba(self, x):
            return [[0.05, 0.95]]

    result = predict_text(Dummy(), "win money now")
    assert result.label in ("spam", "ham")
    assert 0.0 <= result.confidence <= 1.0