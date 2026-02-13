"""FastAPI service for the spam detector.

Meets the ML project extra requirement (Variation B): expose the models via an API
and choose which model to run via a text configuration file.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from spam_detector.config import load_config
from spam_detector.data import load_spam_dataset, train_test_split_dataset
from spam_detector.models import build_pipeline


CONFIG_PATH = Path("config.yaml")
MODEL_DIR = Path("model")
DATA_PATH = Path("data/spam.csv")


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str


def _model_path(model_name: str) -> Path:
    return MODEL_DIR / f"spam_model_{model_name}.joblib"


def _load_or_train(model_name: str):
    """Load cached model; if missing, train quickly on the local dataset."""

    path = _model_path(model_name)
    if path.exists():
        return joblib.load(path)

    # Train & cache
    cfg = load_config(CONFIG_PATH)
    dataset = load_spam_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split_dataset(
        dataset, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model = build_pipeline(model_name, tfidf=cfg.tfidf)
    model.fit(x_train, y_train)

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return model


app = FastAPI(title="Spam Detector API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="text must not be empty")

    cfg = load_config(CONFIG_PATH)
    model = _load_or_train(cfg.model)

    pred = model.predict([text])[0]
    return PredictResponse(label=str(pred))
