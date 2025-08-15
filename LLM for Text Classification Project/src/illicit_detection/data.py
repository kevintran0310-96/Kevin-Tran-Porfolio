"""Dataset loading and preprocessing utilities.

This module provides functions for loading the illicit content
classification datasets, performing train/validation/test splits and
computing class weights.  The functions are intentionally
framework‑agnostic: classical ML models use TF‑IDF features via a
vectorizer, whereas LLMs operate on raw text.
"""

from __future__ import annotations

import logging
import os
from typing import Tuple, Optional, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def load_dataset(path: str) -> pd.DataFrame:
    """Load a CSV dataset with `text` and `label` columns.

    Parameters
    ----------
    path:
        Path to the CSV file.  The CSV must contain at least
        `text` and `label` columns.  Additional columns will be
        preserved but ignored by the training pipeline.

    Returns
    -------
    pandas.DataFrame
        The loaded dataset.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    df = pd.read_csv(path)
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns")
    logger.info("Loaded %d records from %s", len(df), path)
    return df


def split_dataset(
    df: pd.DataFrame,
    train_size: float = 0.8,
    val_size: float = 0.1,
    test_size: float = 0.1,
    random_seed: int = 1337,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into train/validation/test sets.

    The split is performed in two stages: first into train and temp
    (val+test), then the temp is split into validation and test.  If
    `stratify` is True the splits preserve the label distribution.

    Parameters
    ----------
    df:
        The full dataset.
    train_size:
        Proportion of the dataset to include in the train split.
    val_size:
        Proportion of the dataset to include in the validation split.
    test_size:
        Proportion of the dataset to include in the test split.
    random_seed:
        Seed used for the random number generator.
    stratify:
        Whether to stratify splits by the `label` column.

    Returns
    -------
    (train_df, val_df, test_df)
        A tuple of DataFrames corresponding to the splits.
    """
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    y = df["label"] if stratify else None
    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        stratify=y,
        random_state=random_seed,
    )
    # Recompute y for the temp split if stratifying
    y_temp = temp_df["label"] if stratify else None
    val_prop = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_prop,
        stratify=y_temp,
        random_state=random_seed,
    )
    logger.info(
        "Split data: %d train, %d val, %d test",
        len(train_df), len(val_df), len(test_df)
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_class_weights(labels: pd.Series) -> Dict[int, float]:
    """Compute inverse frequency class weights.

    Parameters
    ----------
    labels:
        A Series of integer class labels.

    Returns
    -------
    dict
        Mapping from class label to weight proportional to
        1/frequency.  The weights are normalised so that the
        average weight is 1.0.
    """
    counts = labels.value_counts().to_dict()
    total = len(labels)
    weights = {cls: total / (len(counts) * freq) for cls, freq in counts.items()}
    logger.debug("Class weights computed: %s", weights)
    return weights
