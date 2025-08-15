<!--
This README documents the R port of the Kaggle happiness and mental health project.  It uses the R ecosystem (tidyverse, caret, xgboost, etc.).  Please retain citations when quoting those results.
-->

# 🧠 Predicting Happiness & Mental Health – R Edition

This repository is the **R implementation** of my solutions to two Kaggle competitions:

1. **Regression Contest 24 S‑1 – Predict happiness level** – The target is a numeric
   `happiness` score.  My original solution ranked **1st out of 230 teams** on the
   leaderboard.
2. **Classification 24‑1 – Predict perfect mental health** – The target is a five‑class
   ordinal score taking values \(-2,-1,0,1,2\).  The submitted model placed **61st of 225
   competitors**.

This R project provides a
reproducible, modular pipeline with **out‑of‑fold stacking**, **config‑driven
experimentation**, **leakage prevention**, and **continuous integration**.  It demonstrates
how to build professional data‑science projects in R while preserving the competitive
performance of the original Kaggle notebooks.

## 📜 Features and improvements

* **Out‑of‑fold stacking** – Base learners (random forests, support vector machines,
  gradient boosting machines) are trained on stratified folds to produce OOF predictions.
  A meta‑model (linear regression for the regression task; multinomial logistic
  regression for the classification task) is trained on these OOF features.  This
  technique reduces overfitting and eliminates data leakage.
* **Ordinal‑aware classification** – The mental health score is treated as an ordered
  outcome.  In addition to multiclass classifiers (`randomForest`, `e1071::svm`,
  `xgboost`), the pipeline includes an *ordinal regression* base learner (a linear
  model whose predictions are rounded to the nearest integer).  Macro‑F1 is used for
  hyperparameter tuning and model selection.
* **Centralised preprocessing** – Functions in `R/data_prep.R` parse heights given as
  ranges (e.g. “165 – 170”), map gender strings to integers, encode categorical
  variables via factors and integer encoding, and impute missing values with medians
  (numeric) or modes (categorical).  The same preprocessing is applied consistently
  across all models.
* **Config‑driven experiments** – All file paths, model hyperparameters, number of
  folds and random seeds are specified in YAML files under `configs/`.  Changing the
  experiment requires editing the YAML rather than the R code.
* **Reproducibility via renv** – A `renv.lock` file (optional) can be created by
  running `renv::snapshot()` after installing dependencies.  All random seeds are set
  from a single value in the configuration file.  Stratified splitting uses
  `caret::createDataPartition` for classification and quantile binning for regression.
* **Tests and CI** – Unit tests using `testthat` verify that preprocessing and
  splitting functions behave correctly.  A GitHub Actions workflow installs R,
  restores dependencies and runs the tests automatically.
* **Clear documentation** – This README describes the data, the modelling strategy and
  the improvements made over the original notebook.  It explains how we avoid
  leakage and how to run the project.

## 🗂 Repository structure

```
mental_health_project_R/
├── R/
│   ├── data_prep.R            # Data loading, cleaning and splitting utilities
│   ├── models.R               # Base and meta model definitions and training
│   ├── metrics.R              # Regression and classification metrics
│   └── train_utils.R          # Helper functions for training and prediction
├── configs/
│   ├── regression.yaml        # Experiment configuration for the happiness task
│   └── classification.yaml    # Experiment configuration for the mental health task
├── data/
│   ├── raw/                   # Original Kaggle CSVs (already included)
│   └── processed/             # Generated splits, OOF predictions and models
├── scripts/
│   ├── run_regression.R       # Rscript entry point to train regression models
│   └── run_classification.R   # Rscript entry point to train classification models
├── tests/
│   ├── test_data.R            # Unit tests for preprocessing and splitting
│   └── test_models.R          # Unit tests for model training functions
├── .github/workflows/ci.yml   # Continuous integration pipeline for R
├── install_packages.R         # Convenience script to install required packages
└── README.md (this file)
```

## 🧪 Quick start

1. **Clone the repository**:

   ```bash
   git clone <THIS_REPO_URL>
   cd mental_health_project_R
   ```

2. **Install R dependencies**.  A script `install_packages.R` is provided to install
   required packages from CRAN.  You may also create a renv environment and restore
   from a lockfile once it is generated.

   ```r
   # in R
   source("install_packages.R")
   # Optionally create a project‑local library and snapshot
   # renv::init(); renv::snapshot()
   ```

3. **Run an experiment**.  Use `Rscript` to run the provided scripts.  By default
   they will read the corresponding YAML configuration and write models and OOF
   predictions to `data/processed/`.

   ```bash
   Rscript scripts/run_regression.R --config configs/regression.yaml --output_dir data/processed
   Rscript scripts/run_classification.R --config configs/classification.yaml --output_dir data/processed
   ```

4. **Submit predictions**.  After training, you can generate predictions for the Kaggle
   test sets and assemble submission files using functions in `R/train_utils.R` or
   write your own script.

## 🛡️ Avoiding data leakage

We take several precautions to prevent information from leaking from the validation or
test data into the training process:

* **Out‑of‑fold generation** – When stacking, each base learner generates predictions
  only on the fold held out for validation.  The meta‑model is trained on these
  predictions; it never sees predictions from a model trained on the same rows.
* **Nested resampling** – If feature selection or hyperparameter tuning is used,
  it is done inside each fold so that the outer loop remains unbiased.
* **Stratified splitting** – For classification we use `caret::createDataPartition`
  with the mental health classes; for regression we create quantile bins to
  approximate stratification.  This ensures that rare classes appear in all folds.
* **Seed control** – The configuration files specify a single `random_state` that is
  used to seed R’s RNG (`set.seed()`), as well as any package‑specific RNGs.

## 📏 Metrics

Regression performance is measured via **root mean squared error (RMSE)**.  For
classification we report **accuracy**, **macro‑F1**, **weighted‑F1** and confusion
matrices.  F1 is used for model selection because the Kaggle evaluation metric for
the mental health competition was F1 score.

## 💼 Contributing

Contributions are welcome!  Please see the `tests/` directory for guidance on how to
write additional tests.  All pull requests must pass the test suite and the GitHub
Actions workflow.

---

© 2025 Quoc Khoa Tran.  Please credit this work if you reuse it.