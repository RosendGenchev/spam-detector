from spam_detector.train import build_model, save_model
from pathlib import Path


def test_build_model_returns_pipeline() -> None:
    model = build_model()
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
    
def test_save_model_creates_file(tmp_path) -> None:
    model = build_model()
    out = tmp_path / "m.joblib"
    save_model(model, out)
    assert out.exists()