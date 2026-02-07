from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import pandas as pd


class DataFormatError(ValueError):
    """Raised when the dataset file is missing required columns or has invalid format."""


@dataclass(frozen=True)
class Dataset:
    texts: pd.Series
    labels: pd.Series


def load_spam_dataset(path: Path) -> Dataset:
    """
    Load spam dataset from CSV.

    Supported formats:
    1) Columns: label, text
    2) Columns: v1, v2 (common 'spam.csv' format) -> renamed to label/text

    Labels must be 'spam' or 'ham' (case-insensitive).
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    try:
            # Try common encodings (some spam.csv files are latin-1 / cp1252)
        read_errors = []
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                df = pd.read_csv(path, encoding=enc)
                break
            except UnicodeDecodeError as exc:
                read_errors.append(f"{enc}: {exc}")
        else:
            raise DataFormatError(
                "Failed to decode CSV. Tried encodings: utf-8, utf-8-sig, latin-1, cp1252. "
                f"Errors: {read_errors}"
        )

    except Exception as exc:
        raise DataFormatError(f"Failed to read CSV: {path}") from exc

    # Handle common "v1/v2" format
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "text"})

    if "label" not in df.columns or "text" not in df.columns:
        raise DataFormatError(
            f"CSV must contain columns 'label' and 'text' (or 'v1'/'v2'). Got: {list(df.columns)}"
        )

    df = df[["label", "text"]].dropna()
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    df["text"] = df["text"].astype(str)

    df = df[df["label"].isin(["spam", "ham"])]

    if df.empty:
        raise DataFormatError("Dataset is empty after filtering for labels {'spam','ham'}.")

    return Dataset(texts=df["text"], labels=df["label"])


def train_test_split_dataset(
    dataset: Dataset, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simple wrapper to keep split logic in one place."""
    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test = train_test_split(
        dataset.texts,
        dataset.labels,
        test_size=test_size,
        random_state=random_state,
        stratify=dataset.labels,
    )
    return x_train, x_test, y_train, y_test
