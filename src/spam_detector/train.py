"""Training pipeline for the spam detector model."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from spam_detector.data import Dataset, load_spam_dataset, train_test_split_dataset


MODEL_PATH = Path("model/spam_model.joblib")
DATA_PATH = Path("data/spam.csv")


def build_model() -> Pipeline:
    """Build and return a TF-IDF + Naive Bayes classification pipeline."""
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(stop_words="english")),
            ("clf", MultinomialNB()),
        ]
    )


def train() -> Tuple[Pipeline, float]:
    """Train the spam detection model and return the trained model and its accuracy."""
    dataset: Dataset = load_spam_dataset(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split_dataset(dataset)

    model: Pipeline = build_model()
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    print("Accuracy:", round(acc, 4))
    print("\nConfusion matrix:\n", confusion_matrix(y_test, preds))
    print("\nClassification report:\n", classification_report(y_test, preds))

    return model, acc


def save_model(model: Pipeline, path: Path) -> None:
    """Save the trained model to disk using joblib."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"✅ Model saved to {path}")



def main() -> None:
    """Train the spam detector model and persist it to disk."""
    model, _ = train()
    model_path = Path("model") / "spam_model_nb.joblib"
    save_model(model, model_path)



if __name__ == "__main__":
    main()
