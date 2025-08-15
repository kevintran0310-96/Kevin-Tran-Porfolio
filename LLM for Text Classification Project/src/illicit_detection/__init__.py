"""Top level package for illicit content classification.

This package exposes a high‑level API for loading data, training
multiple model types and performing inference on new text.  The
implementation details are organised into submodules to facilitate
testing and extensibility.
"""

from .data import load_dataset, split_dataset, get_class_weights
from .metrics import compute_metrics
from .train import main as train_cli
from .infer import main as infer_cli

__all__ = [
    "load_dataset",
    "split_dataset",
    "get_class_weights",
    "compute_metrics",
    "train_cli",
    "infer_cli",
]