from pathlib import Path

from spam_detector.data import load_spam_dataset


def test_load_spam_dataset_has_expected_labels() -> None:
    ds = load_spam_dataset(Path("data/spam.csv"))
    labels = set(ds.labels.unique())
    assert "spam" in labels
    assert "ham" in labels
    assert len(ds.texts) == len(ds.labels)
    assert len(ds.texts) > 100  # sanity check
