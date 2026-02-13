"""Training pipeline for the spam detector model.

Trains a selected model (Naive Bayes or Logistic Regression) using a TF-IDF pipeline
and saves it to disk so it can be used by the CLI and the FastAPI service.
"""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from spam_detector.config import load_config
from spam_detector.data import load_spam_dataset, train_test_split_dataset
from spam_detector.models import ModelName, TfidfConfig, build_pipeline

CONFIG_PATH = Path("config.yaml")
DATA_PATH = Path("data/spam.csv")
MODEL_DIR = Path("model")


def build_model(model: ModelName = "nb", tfidf: TfidfConfig | None = None) -> Pipeline:
    """Build a model pipeline (used by tests too)."""
    return build_pipeline(model, tfidf=tfidf)


def train_and_evaluate() -> tuple[Pipeline, float]:
    """Train the configured model and return (model, accuracy on the test split)."""
    cfg = load_config(CONFIG_PATH)

    dataset = load_spam_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split_dataset(
        dataset, test_size=cfg.test_size, random_state=cfg.random_state
    )

    model = build_pipeline(cfg.model, tfidf=cfg.tfidf)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = float(accuracy_score(y_test, preds))

    print(f"Configured model: {cfg.model}")
    print("Accuracy:", round(acc, 4))
    print("Confusion matrix:", confusion_matrix(y_test, preds))
    print("Classification report:\n", classification_report(y_test, preds))

    return model, acc


def save_model(model: Pipeline, path: Path) -> Path:
    """Save the model to the given path and return it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def main() -> None:
    """Train the model and save it to model/spam_model_<model>.joblib."""
    model, _ = train_and_evaluate()
    cfg = load_config(CONFIG_PATH)

    out_path = MODEL_DIR / f"spam_model_{cfg.model}.joblib"
    save_model(model, out_path)

    print(f"✅ Model saved to {out_path}")


if __name__ == "__main__":
    main()
