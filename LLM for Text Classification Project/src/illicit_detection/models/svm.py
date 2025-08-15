"""Support Vector Machine baseline for text classification."""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from sklearn.svm import LinearSVC


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    class_weights: Optional[Dict[int, float]] = None,
    C: float = 1.0,
    kernel: str = "linear",
) -> LinearSVC:
    """Train a linear SVM classifier.

    Parameters
    ----------
    X_train:
        Feature matrix (documents × features).
    y_train:
        Array of class labels.
    class_weights:
        Optional dictionary mapping class label to weight.  Useful for
        handling class imbalance.
    C:
        Regularisation parameter.
    kernel:
        Kernel type.  Only ``"linear"`` is supported for computational
        efficiency.

    Returns
    -------
    LinearSVC
        A trained linear SVM model.
    """
    if kernel != "linear":
        raise ValueError("Only linear kernel is supported")
    svm = LinearSVC(C=C, class_weight=class_weights)
    svm.fit(X_train, y_train)
    return svm
