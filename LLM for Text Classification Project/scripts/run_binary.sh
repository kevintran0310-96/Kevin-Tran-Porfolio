#!/usr/bin/env bash
set -e

# Example script for running the binary classification task with various models.

python -m illicit_detection.train --config configs/binary.yaml --model svm
python -m illicit_detection.train --config configs/binary.yaml --model naive_bayes
python -m illicit_detection.train --config configs/binary.yaml --model bert
python -m illicit_detection.train --config configs/binary.yaml --model llama
python -m illicit_detection.train --config configs/binary.yaml --model gemma
