"""Inference script for illicit content models.

Given a trained model and a piece of text, this script will produce a
prediction and optionally the top‑k probabilities.  Classical models
require a separate TF‑IDF vectorizer file, whereas neural models
loaded via HuggingFace do not.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np


def load_classical_model(model_path: Path, vectorizer_path: Optional[Path]):
    import joblib
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path) if vectorizer_path is not None else None
    return model, vectorizer


def predict_classical(model, vectorizer, text: str) -> int:
    X = vectorizer.transform([text])
    return int(model.predict(X)[0])


def load_llm_model(model_path: Path):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    return tokenizer, model


def predict_llm(tokenizer, model, text: str, topk: int = 3):
    import torch
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    top_indices = probs.argsort()[::-1][:topk]
    return top_indices.tolist(), probs[top_indices].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference on illicit content models")
    parser.add_argument("--model_path", required=True, help="Path to saved model (.joblib or HuggingFace directory)")
    parser.add_argument("--vectorizer_path", help="Path to TF‑IDF vectorizer (.joblib) for classical models")
    parser.add_argument("--text", required=True, help="The input text to classify")
    parser.add_argument("--topk", type=int, default=3, help="Number of top probabilities to return for LLMs")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    vectorizer_path = Path(args.vectorizer_path) if args.vectorizer_path else None

    if model_path.suffix == ".joblib":
        model, vectorizer = load_classical_model(model_path, vectorizer_path)
        label = predict_classical(model, vectorizer, args.text)
        print(json.dumps({"label": label}))
    else:
        tokenizer, model = load_llm_model(model_path)
        top_indices, top_probs = predict_llm(tokenizer, model, args.text, args.topk)
        print(json.dumps({"top_labels": top_indices, "probs": top_probs}))


if __name__ == "__main__":
    main()
