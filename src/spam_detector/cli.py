"""Command-line interface for the spam detector."""
from __future__ import annotations

from pathlib import Path

from spam_detector.config import load_config
from spam_detector.predict import ModelNotFoundError, load_model, predict_text

MODEL_PATH = Path("model/spam_model.joblib")
CONFIG_PATH = Path("config.yaml")

def main() -> None:
    """Run an interactive CLI that predicts spam/ham for user-entered messages."""
    print("Spam Detector (type empty line to exit)\n")

    # If config.yaml exists, prefer the model selected there.
    model_path = MODEL_PATH
    if CONFIG_PATH.exists():
        try:
            cfg = load_config(CONFIG_PATH)
            model_path = Path("model") / f"spam_model_{cfg.model}.joblib"
        except Exception:
            # Keep fallback path if config is invalid
            model_path = MODEL_PATH

    try:
        model = load_model(model_path)
    except ModelNotFoundError as exc:
        print(f"❌ {exc}")
        return

    while True:
        try:
            text = input("Enter message: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            return

        if text == "":
            print("Bye!")
            return

        try:
            result = predict_text(model, text)
            print(f"➡️  {result.label.upper()}  (confidence: {result.confidence:.2f})\n")
        except ValueError as exc:
            print(f"⚠️  {exc}\n")


if __name__ == "__main__":
    main()
