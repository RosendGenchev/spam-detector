"""Command-line interface for the spam detector."""
from __future__ import annotations

from pathlib import Path

from spam_detector.predict import ModelNotFoundError, load_model, predict_text

MODEL_PATH = Path("model/spam_model.joblib")

def main() -> None:
    """Run an interactive CLI that predicts spam/ham for user-entered messages."""
    print("Spam Detector (type empty line to exit)\n")

    try:
        model = load_model(MODEL_PATH)
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
