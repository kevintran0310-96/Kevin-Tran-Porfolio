"""Model registry for illicit content detection.

Each model module should expose a `train()` function that returns
metrics and a trained model given the training, validation and test
datasets.  Classical models may also expose additional helpers.
"""

from .svm import train as train_svm
from .naive_bayes import train as train_naive_bayes
try:
    from .bert import train as train_bert
except Exception:  # pragma: no cover
    train_bert = None
try:
    from .llama import train as train_llama
except Exception:
    train_llama = None
try:
    from .gemma import train as train_gemma
except Exception:
    train_gemma = None

__all__ = [
    "train_svm",
    "train_naive_bayes",
    "train_bert",
    "train_llama",
    "train_gemma",
]