#!/usr/bin/env bash
set -e

# Example script for running the multi‑class classification task with various models.

python -m illicit_detection.train --config configs/multiclass.yaml --model svm
python -m illicit_detection.train --config configs/multiclass.yaml --model naive_bayes
python -m illicit_detection.train --config configs/multiclass.yaml --model bert
python -m illicit_detection.train --config configs/multiclass.yaml --model llama
python -m illicit_detection.train --config configs/multiclass.yaml --model gemma
