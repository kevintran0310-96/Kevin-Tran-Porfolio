"""Command‑line interface for training models.

This script ties together the data loading, model training and
evaluation components.  It supports classical baselines (SVM, Naive
Bayes) and LLMs (BERT, Llama, Gemma) via a unified interface.  All
hyperparameters and file paths are supplied via a YAML config file.

Usage:

```bash
python -m illicit_detection.train --config configs/binary.yaml --model svm
```

The trained model and any associated vectorizers or checkpoints are
saved under the directory specified in the config (see
`model_dir`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, Tuple

import yaml
import numpy as np

from .data import load_dataset, split_dataset, get_class_weights
from .metrics import compute_metrics


def set_seed(seed: int) -> None:
    """Set global random seeds for reproducibility."""
    import random
    import numpy as np  # ensure local import for seeding
    try:
        import torch
    except ImportError:
        torch = None
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_config(path: str) -> Dict[str, any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_svm_pipeline(config: Dict[str, any], train_df, val_df, test_df) -> Tuple[Dict[str, float], any, any]:
    """Train and evaluate an SVM model.

    Returns metrics on the test set and the trained model/vectorizer.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .models.svm import train as train_svm
    # Vectorize text
    vec_cfg = config.get("svm", {})
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
        min_df=vec_cfg.get("min_df", 2),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    # Compute class weights
    class_weights = get_class_weights(train_df["label"])
    # Train model
    model = train_svm(X_train, train_df["label"].to_numpy(), class_weights=class_weights, C=vec_cfg.get("C", 1.0), kernel=vec_cfg.get("kernel", "linear"))
    # Evaluate
    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_df["label"], y_pred)
    return metrics, model, vectorizer


def train_naive_bayes_pipeline(config: Dict[str, any], train_df, val_df, test_df) -> Tuple[Dict[str, float], any, any]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from .models.naive_bayes import train as train_nb
    vec_cfg = config.get("naive_bayes", {})
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(vec_cfg.get("ngram_range", [1, 2])),
        min_df=vec_cfg.get("min_df", 2),
    )
    X_train = vectorizer.fit_transform(train_df["text"])
    X_test = vectorizer.transform(test_df["text"])
    class_weights = None  # NB cannot handle class weights easily
    model = train_nb(X_train, train_df["label"].to_numpy(), alpha=vec_cfg.get("alpha", 1.0))
    y_pred = model.predict(X_test)
    metrics = compute_metrics(test_df["label"], y_pred)
    return metrics, model, vectorizer


def train_bert_pipeline(config: Dict[str, any], train_df, val_df, test_df, task: str) -> Tuple[Dict[str, float], any]:
    from .models.bert import train as train_bert
    # Convert DataFrames to lists of dicts for HuggingFace dataset
    train_dataset = {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()}
    val_dataset = {"text": val_df["text"].tolist(), "label": val_df["label"].tolist()}
    test_dataset = {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}
    metrics, model = train_bert(
        train_dataset,
        val_dataset,
        test_dataset,
        num_labels=train_df["label"].nunique(),
        config=config,
        task=task,
    )
    return metrics, model


def train_llama_pipeline(config: Dict[str, any], train_df, val_df, test_df, task: str) -> Tuple[Dict[str, float], any]:
    from .models.llama import train as train_llama
    train_dataset = {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()}
    val_dataset = {"text": val_df["text"].tolist(), "label": val_df["label"].tolist()}
    test_dataset = {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}
    metrics, model = train_llama(
        train_dataset,
        val_dataset,
        test_dataset,
        num_labels=train_df["label"].nunique(),
        config=config,
        task=task,
    )
    return metrics, model


def train_gemma_pipeline(config: Dict[str, any], train_df, val_df, test_df, task: str) -> Tuple[Dict[str, float], any]:
    from .models.gemma import train as train_gemma
    train_dataset = {"text": train_df["text"].tolist(), "label": train_df["label"].tolist()}
    val_dataset = {"text": val_df["text"].tolist(), "label": val_df["label"].tolist()}
    test_dataset = {"text": test_df["text"].tolist(), "label": test_df["label"].tolist()}
    metrics, model = train_gemma(
        train_dataset,
        val_dataset,
        test_dataset,
        num_labels=train_df["label"].nunique(),
        config=config,
        task=task,
    )
    return metrics, model


def save_model(model, vectorizer, model_dir: Path, model_name: str) -> None:
    """Persist trained model and vectorizer to disk."""
    model_dir.mkdir(parents=True, exist_ok=True)
    # Classical models use joblib
    try:
        import joblib
        joblib.dump(model, model_dir / f"{model_name}.joblib")
        if vectorizer is not None:
            joblib.dump(vectorizer, model_dir / f"{model_name}_vectorizer.joblib")
    except Exception:
        # For deep models we rely on huggingface save_pretrained
        try:
            model.save_pretrained(str(model_dir / model_name))
        except Exception as e:
            logging.error("Failed to save model: %s", e)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train models for illicit content detection")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--model", required=True, choices=["svm", "naive_bayes", "bert", "llama", "gemma"], help="Which model to train")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    cfg = load_config(args.config)
    set_seed(cfg.get("random_seed", 1337))

    dataset_path = cfg["dataset_path"]
    task = cfg.get("task", "binary")
    df = load_dataset(dataset_path)
    train_df, val_df, test_df = split_dataset(
        df,
        train_size=cfg.get("train_size", 0.8),
        val_size=cfg.get("val_size", 0.1),
        test_size=cfg.get("test_size", 0.1),
        random_seed=cfg.get("random_seed", 1337),
        stratify=True,
    )

    model_dir = Path(cfg.get("model_dir", "models"))
    model_name = args.model

    if args.model == "svm":
        metrics, model, vec = train_svm_pipeline(cfg, train_df, val_df, test_df)
        save_model(model, vec, model_dir, model_name)
    elif args.model == "naive_bayes":
        metrics, model, vec = train_naive_bayes_pipeline(cfg, train_df, val_df, test_df)
        save_model(model, vec, model_dir, model_name)
    elif args.model == "bert":
        metrics, model = train_bert_pipeline(cfg, train_df, val_df, test_df, task)
        save_model(model, None, model_dir, model_name)
    elif args.model == "llama":
        metrics, model = train_llama_pipeline(cfg, train_df, val_df, test_df, task)
        save_model(model, None, model_dir, model_name)
    elif args.model == "gemma":
        metrics, model = train_gemma_pipeline(cfg, train_df, val_df, test_df, task)
        save_model(model, None, model_dir, model_name)
    else:
        raise ValueError(f"Unknown model: {args.model}")

    # Print metrics
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
