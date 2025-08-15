"""Tests for the data module."""

import pandas as pd
from illicit_detection.data import split_dataset, get_class_weights


def test_split_dataset_stratified():
    # Synthetic dataset with imbalanced classes
    df = pd.DataFrame({
        "text": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        "label": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })
    train, val, test = split_dataset(df, train_size=0.6, val_size=0.2, test_size=0.2, random_seed=42)
    # Check sizes
    assert len(train) == 6
    assert len(val) == 2
    assert len(test) == 2
    # Check approximate stratification: proportion of class 1 should be 0.5
    assert abs(train["label"].mean() - 0.5) < 0.2
    assert abs(val["label"].mean() - 0.5) < 0.5  # small sample
    assert abs(test["label"].mean() - 0.5) < 0.5


def test_class_weights():
    labels = pd.Series([0, 0, 0, 1])
    weights = get_class_weights(labels)
    # Class 0 has freq 3, class 1 has freq 1
    assert weights[1] > weights[0]
