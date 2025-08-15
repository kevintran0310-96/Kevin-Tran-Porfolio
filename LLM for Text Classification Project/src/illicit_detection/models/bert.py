"""BERT fine‑tuning for text classification.

This module implements a training routine for BERT on either the
binary or multi‑class illicit content detection task.  It uses the
HuggingFace Transformers `Trainer` API for convenience.  The default
model is `bert-base-uncased`, but this can be overridden via the
configuration file.
"""

from __future__ import annotations

import logging
from typing import Dict, Tuple, Any

import numpy as np

try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        Trainer,
        TrainingArguments,
        DataCollatorWithPadding,
    )
    from datasets import Dataset
    from .metrics import compute_metrics
except Exception as e:  # pragma: no cover
    # If transformers/datasets not installed, training will fail gracefully.
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Trainer = None
    TrainingArguments = None
    DataCollatorWithPadding = None
    Dataset = None
    compute_metrics = None
    logging.warning("Transformers or Datasets library not available: %s", e)


def train(
    train_dataset: Dict[str, list],
    val_dataset: Dict[str, list],
    test_dataset: Dict[str, list],
    num_labels: int,
    config: Dict[str, Any],
    task: str,
) -> Tuple[Dict[str, float], Any]:
    """Fine‑tune BERT on the illicit content detection dataset.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset:
        Dictionaries with keys ``text`` and ``label``.  These will be
        converted into HuggingFace Datasets.
    num_labels:
        Number of distinct classes.
    config:
        A dictionary of hyperparameters loaded from YAML.
    task:
        Either ``"binary"`` or ``"multiclass"``; currently unused but
        kept for parity with other models.

    Returns
    -------
    (metrics, model)
        A tuple containing the metrics on the test set and the
        fine‑tuned model.
    """
    if AutoTokenizer is None or Dataset is None:
        raise ImportError("transformers and datasets must be installed to train BERT models")
    # Load tokenizer and model
    model_name = config.get("bert_model_name", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    # Convert raw dicts into HF Datasets
    train_ds = Dataset.from_dict(train_dataset)
    val_ds = Dataset.from_dict(val_dataset)
    test_ds = Dataset.from_dict(test_dataset)
    # Tokenize
    def preprocess(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=config.get("max_seq_length", 128),
        )
    train_ds = train_ds.map(preprocess, batched=True)
    val_ds = val_ds.map(preprocess, batched=True)
    test_ds = test_ds.map(preprocess, batched=True)
    # Data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.get("model_dir", "models"),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 32),
        per_device_eval_batch_size=config.get("batch_size", 32),
        learning_rate=config.get("learning_rate", 2e-5),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=config.get("log_dir", "runs"),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )
    # Wrap compute_metrics for Trainer
    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds)
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
    )
    # Train
    trainer.train()
    # Evaluate on test set
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    return test_metrics, model
