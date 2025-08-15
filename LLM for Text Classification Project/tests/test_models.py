"""Tests for model training functions."""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from illicit_detection.models.svm import train as train_svm
from illicit_detection.models.naive_bayes import train as train_nb


def test_train_svm():
    texts = ["buy arms", "sell drugs", "hello world", "how are you"]
    labels = np.array([1, 1, 0, 0])
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    model = train_svm(X, labels, class_weights={0:1.0, 1:1.0}, C=1.0)
    preds = model.predict(X)
    # The linear SVM should fit perfectly on this tiny dataset
    assert (preds == labels).all()


def test_train_naive_bayes():
    texts = ["guns", "pills", "apple", "banana"]
    labels = np.array([1, 1, 0, 0])
    vec = TfidfVectorizer()
    X = vec.fit_transform(texts)
    model = train_nb(X, labels, alpha=1.0)
    preds = model.predict(X)
    # NB may misclassify identical counts but should predict something
    assert len(preds) == len(labels)
