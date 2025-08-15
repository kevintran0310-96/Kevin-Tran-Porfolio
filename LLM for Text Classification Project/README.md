# LLM‑Powered Illicit Content Detection

This repository contains a **production‑ready reimplementation** of the research project
“Using LLM to detect illicit content on online marketplaces.”  The original
work explored a suite of models—from classic machine learning baselines
to state‑of‑the‑art large language models (LLMs)—to identify harmful
listings in a multilingual dataset.  This refactor transforms that
exploratory code into a clean, reproducible package suitable for
industry use and hiring portfolios.

## 📊 TL;DR Results

| Task               | Best model     | Accuracy | Macro‑F1 | Notes |
|--------------------|---------------|:-------:|:-------:|------|
| **Binary** (illicit vs non‑illicit) | **SVM**  | **0.90** | **0.81** | Classic linear SVM with TF‑IDF features slightly outperforms Llama 3.2 despite vastly smaller size【144†L1-L6】. |
| **Multi‑class** (40 illicit categories) | **Llama 3.2** | **0.74** | **0.61** | Llama’s semantic understanding gives it a clear edge on the complex multi‑class task【144†L1-L6】. |

Complete performance tables and experimental details can be found in
the original research paper (`/Using LLM to detect illicit content (final) copy.pdf`) and in the
notebooks under `/reports`.  This repository summarises the key
insights and provides code to reproduce the experiments end‑to‑end.

## 🗂 Repository layout

```
illicit_detection_project/
├── README.md                # Project overview (this file)
├── requirements.txt         # Pip dependencies
├── environment.yml          # Conda environment definition
├── labels.json              # List of illicit categories (40)
├── configs/
│   ├── binary.yaml          # Hyperparameters & paths for binary task
│   └── multiclass.yaml      # Hyperparameters & paths for multi‑class task
├── src/illicit_detection/
│   ├── __init__.py
│   ├── data.py              # Dataset loading & preprocessing
│   ├── metrics.py           # Common evaluation metrics
│   ├── train.py             # CLI for training & evaluation
│   ├── infer.py             # CLI for inference on new text
│   └── models/
│       ├── __init__.py
│       ├── svm.py
│       ├── naive_bayes.py
│       ├── bert.py
│       ├── llama.py
│       └── gemma.py
├── scripts/
│   ├── run_binary.sh        # Example run script for binary classification
│   └── run_multiclass.sh    # Example run script for multi‑class classification
├── tests/                   # Unit tests (small synthetic datasets)
│   ├── test_data.py
│   └── test_models.py
└── .github/workflows/ci.yml # Continuous integration (linting & tests)
```

### Key improvements

* **CLI training interface.**  Experiments are controlled via a single
  `train.py` script with command‑line flags and YAML configs.  You no
  longer need to step through notebooks to run an experiment.
* **Modular architecture.**  Each algorithm lives in its own module
  under `src/illicit_detection/models/` with a unified API.  Adding
  another model is as simple as implementing a `train()` function.
* **Reproducible splits.**  Data loading functions perform
  deterministic stratified train/val/test splits and return class
  weights to counter class imbalance.
* **Configuration files.**  All file paths and hyperparameters are
  externalised into YAML files.  You can run the same experiment with
  different settings by editing a single config file.
* **Seed control.**  A global random seed is set in one place to
  ensure deterministic behaviour across runs.
* **Tests & CI.**  Unit tests on toy datasets validate that the
  high‑level training pipeline functions correctly.  A GitHub Action
  runs these tests automatically on every commit.
* **Baseline & LLM parity.**  Classical models (SVM, Naive Bayes)
  share the same preprocessing pipeline as the LLMs (BERT, Llama,
  Gemma), allowing apples‑to‑apples comparisons and ablation studies.

## 🚀 Getting started

### 1. Clone the repository

```bash
git clone <your‑fork‑url>
cd illicit_detection_project
```

### 2. Install dependencies

This project assumes a modern Python environment with GPU support for
the LLMs.  Two options are provided:

#### Using conda

```bash
conda env create -f environment.yml
conda activate illicit
```

#### Using pip

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Prepare the dataset

The experiments expect a CSV file with two columns:

* `text` – the listing title or description.
* `label` – the class label (``0``/``1`` for binary or one of the 40
  classes for multi‑class).  The full list of classes is provided in
  `labels.json`.

Place your dataset at the path specified in the config file (see
`configs/binary.yaml` or `configs/multiclass.yaml`).  If you’re
using the DUTA10K dataset from the accompanying paper, you can
preprocess it into this format using a short script (not included).

### 4. Run an experiment

To train and evaluate a model specified in a config file, run:

```bash
python -m illicit_detection.train \
  --config configs/binary.yaml \
  --model svm
```

This will load the dataset, perform a train/validation/test split,
train the specified model on the training set, evaluate on the
validation set at the end of each epoch (if applicable), and report
final metrics on the test set.  Metrics such as accuracy, macro F1,
weighted F1, precision and recall are printed to stdout.

Example run scripts are provided under `scripts/`; you can modify
these or invoke `train.py` directly with your own parameters.

### 5. Inference on new text

Once you have a trained model saved to disk (the default output
location is `models/{model_name}.pt` for neural models or `.joblib`
for classical ones), you can perform inference on new strings:

```bash
python -m illicit_detection.infer \
  --model_path models/svm.joblib \
  --vectorizer_path models/svm_vectorizer.joblib \
  --text "Buy bitcoin quickly"
```

This will print the predicted label and the top‑k class probabilities.

## 🔬 Research notes & additional materials

The original research paper (in PDF format) provides a deep dive
into the methodology, data preparation and experimental results.  You
can find it in the root of this repository.  Key findings include:

* **Classic models punch above their weight.**  With a strong TF‑IDF
  pipeline and careful tuning, a linear SVM achieves 0.90 accuracy
  and 0.81 macro F1 on the binary task, slightly outperforming
  Llama 3.2【144†L1-L6】.
* **LLMs excel at fine‑grained classification.**  For the 40‑class
  multi‑class task, Llama 3.2 reaches 0.74 accuracy and 0.61 macro F1,
  significantly ahead of both Gemma 3 and BERT【144†L1-L6】.
* **Imbalanced data requires care.**  Weighted loss functions and
  class weighting are essential to prevent minority classes from being
  ignored.  Our pipeline computes class weights automatically and
  passes them to the training routine.

Please see the PDF for the full literature review, data ethics
discussion and ablation studies.

## 📜 License & ethical considerations

This repository is provided for research and educational use only.
The models trained here are intended to assist moderation efforts but
are **not a substitute for human review**.  Misclassification of
illicit or innocuous content can have serious consequences, and
dataset biases may amplify existing prejudices.  Use responsibly and
at your own risk.
