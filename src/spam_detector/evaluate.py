"""Model evaluation and visualization.

Creates plots to satisfy the ML project requirement:
- evaluate model on test data using suitable metrics
- visualize results with plots (confusion matrix, model comparison)

Run:
  python -m spam_detector.evaluate
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless backend (no tkinter needed)

import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from spam_detector.config import load_config
from spam_detector.data import load_spam_dataset, train_test_split_dataset
from spam_detector.models import ModelName, build_pipeline


DATA_PATH = Path("data/spam.csv")
CONFIG_PATH = Path("config.yaml")
REPORTS_DIR = Path("reports")


def _ensure_reports_dir() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_one(model_name: ModelName, x_train, y_train, x_test, y_test, *, tfidf):
    model = build_pipeline(model_name, tfidf=tfidf)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    return {
        "name": model_name,
        "model": model,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, pos_label="spam")),
        "recall": float(recall_score(y_test, preds, pos_label="spam")),
        "f1": float(f1_score(y_test, preds, pos_label="spam")),
        "preds": preds,
    }


def plot_confusion_matrix(y_test, preds, title: str, filename: str) -> None:
    _ensure_reports_dir()
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=160)
    plt.close()


def plot_model_comparison(results, filename: str) -> None:
    _ensure_reports_dir()
    names = [r["name"] for r in results]
    f1s = [r["f1"] for r in results]

    plt.figure()
    plt.bar(names, f1s)
    plt.title("Model comparison (F1 for spam)")
    plt.ylabel("F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / filename, dpi=160)
    plt.close()


def main() -> None:
    cfg = load_config(CONFIG_PATH)

    dataset = load_spam_dataset(DATA_PATH)
    x_train, x_test, y_train, y_test = train_test_split_dataset(
        dataset, test_size=cfg.test_size, random_state=cfg.random_state
    )

    # Baseline: always predict the most common class
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(x_train, y_train)
    dummy_preds = dummy.predict(x_test)
    dummy_f1 = float(f1_score(y_test, dummy_preds, pos_label="spam"))

    results = [
        {
            "name": "dummy",
            "f1": dummy_f1,
            "preds": dummy_preds,
        },
        evaluate_one("nb", x_train, y_train, x_test, y_test, tfidf=cfg.tfidf),
        evaluate_one("logreg", x_train, y_train, x_test, y_test, tfidf=cfg.tfidf),
    ]

    # Print detailed report for the configured model
    chosen = next(r for r in results if r["name"] == cfg.model)
    print(f"\nConfigured model: {cfg.model}\n")
    print("Classification report:\n")
    print(classification_report(y_test, chosen["preds"]))

    plot_confusion_matrix(
        y_test,
        chosen["preds"],
        title=f"Confusion matrix ({cfg.model})",
        filename=f"confusion_matrix_{cfg.model}.png",
    )

    # Comparison plot (F1)
    plot_model_comparison(results, filename="model_comparison_f1.png")

    print(f"\n✅ Plots saved in: {REPORTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
