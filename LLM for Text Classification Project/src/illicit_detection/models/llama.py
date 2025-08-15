"""Llama fine‑tuning with LoRA for illicit content detection.

This module implements a training routine for Meta Llama models
adapted to sequence classification tasks.  It leverages Parameter
Efficient Fine‑Tuning (PEFT) via LoRA to reduce memory footprint
and bitsandbytes quantisation for 4‑bit models.  The code is
written to be illustrative; running it requires a GPU with
sufficient VRAM and the necessary libraries installed.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Tuple

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
    from peft import LoraConfig, get_peft_model
    from .metrics import compute_metrics
except Exception as e:  # pragma: no cover
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None
    Trainer = None
    TrainingArguments = None
    DataCollatorWithPadding = None
    Dataset = None
    LoraConfig = None
    get_peft_model = None
    compute_metrics = None
    logging.warning("LLama training dependencies missing: %s", e)


def train(
    train_dataset: Dict[str, list],
    val_dataset: Dict[str, list],
    test_dataset: Dict[str, list],
    num_labels: int,
    config: Dict[str, Any],
    task: str,
) -> Tuple[Dict[str, float], Any]:
    """Fine‑tune a Llama model with LoRA.

    Parameters
    ----------
    train_dataset, val_dataset, test_dataset:
        Dictionaries with keys ``text`` and ``label``.
    num_labels:
        Number of distinct classes.
    config:
        Hyperparameter dictionary.  Expects keys like
        ``lora_rank``, ``lora_alpha``, ``lora_dropout`` and
        ``quantisation_bits``.
    task:
        Either ``binary`` or ``multiclass``.

    Returns
    -------
    (metrics, model)
        Test set metrics and the fine‑tuned model.
    """
    if AutoTokenizer is None or Dataset is None or LoraConfig is None:
        raise ImportError("Required libraries for Llama training are not installed")
    model_name = config.get("llama_model_name", "meta-llama/Llama-3-8b-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Load base model for classification.  Use sequence classification
    # variant rather than causal LM to simplify classification training.
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        load_in_4bit=(config.get("quantisation_bits", 4) == 4),
        device_map="auto",
    )
    # Configure LoRA
    lora_config = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        bias="none",
        task_type="SEQ_CLS",
    )
    model = get_peft_model(model, lora_config)
    # Convert raw data to HF Dataset
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
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    training_args = TrainingArguments(
        output_dir=config.get("model_dir", "models"),
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("batch_size", 4),
        per_device_eval_batch_size=config.get("batch_size", 4),
        learning_rate=config.get("learning_rate", 2e-5),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir=config.get("log_dir", "runs"),
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
    )
    # Metrics wrapper
    def hf_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return compute_metrics(labels, preds)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=hf_metrics,
    )
    trainer.train()
    test_metrics = trainer.evaluate(eval_dataset=test_ds)
    return test_metrics, model
