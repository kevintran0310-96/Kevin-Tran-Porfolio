"""Evaluation metrics for classification models.

This module provides wrappers around scikit‑learn’s metric
implementations to compute accuracy, macro F1, weighted F1 and other
statistics.  All functions return floats and can be composed into a
single `compute_metrics` function expected by the HuggingFace
`Trainer` API.
"""

from __future__ import annotations

from typing import Dict, Any, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_metrics(y_true: List[int] | np.ndarray, y_pred: List[int] | np.ndarray) -> Dict[str, float]:
    """Compute a set of standard metrics for multi‑class classification.

    Parameters
    ----------
    y_true:
        True class labels.
    y_pred:
        Predicted class labels.

    Returns
    -------
    dict
        Dictionary containing accuracy, macro F1 and weighted F1.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
    }


def confusion_matrix_report(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Return the confusion matrix as a 2D array."""
    return confusion_matrix(y_true, y_pred)
