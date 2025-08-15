"""Naive Bayes baseline for text classification."""

from __future__ import annotations

import numpy as np
from sklearn.naive_bayes import MultinomialNB


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float = 1.0,
) -> MultinomialNB:
    """Train a Multinomial Naive Bayes classifier.

    Parameters
    ----------
    X_train:
        Feature matrix.
    y_train:
        Array of class labels.
    alpha:
        Additive smoothing parameter.

    Returns
    -------
    MultinomialNB
        A trained Naive Bayes classifier.
    """
    model = MultinomialNB(alpha=alpha)
    model.fit(X_train, y_train)
    return model
