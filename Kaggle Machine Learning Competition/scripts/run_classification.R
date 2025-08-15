#!/usr/bin/env Rscript

## Script to train classification models for the mental health prediction task.
##
## Usage:
##   Rscript scripts/run_classification.R --config configs/classification.yaml --output_dir data/processed

suppressPackageStartupMessages({
  library(yaml)
})

## The working directory should be the project root.  Source the R modules from the R folder.
source(file.path("R", "data_prep.R"))
source(file.path("R", "models.R"))
source(file.path("R", "metrics.R"))
source(file.path("R", "train_utils.R"))

args <- commandArgs(trailingOnly = TRUE)
args_list <- list()
for (i in seq(1, length(args), by = 2)) {
  key <- sub("^--", "", args[[i]])
  value <- args[[i + 1]]
  args_list[[key]] <- value
}

config_path <- args_list$config
output_dir <- args_list$output_dir %||% "data/processed"

if (is.null(config_path)) {
  stop("--config argument is required")
}
cfg <- yaml::read_yaml(config_path)
set.seed(cfg$random_state)

# Load and preprocess training data
raw_df <- load_dataset(cfg$train_path)
prep <- preprocess_data(raw_df, target_col = cfg$target_column, task = "classification")
X <- prep$X
y <- prep$y

class_labels <- cfg$class_labels

# Create folds
folds <- create_cv_splits(X, y, n_splits = cfg$n_splits, random_state = cfg$random_state, stratify = TRUE)

# Train base models and meta model
train_res <- train_base_models_classification(
  X = X,
  y = y,
  model_names = cfg$base_models,
  model_params = cfg$model_params,
  folds = folds
)
oof <- train_res$oof
base_models <- train_res$fitted_models

# Round ordinal_reg predictions in OOF
ordinal_indices <- which(cfg$base_models == "ordinal_reg")
if (length(ordinal_indices) > 0) {
  for (idx in ordinal_indices) {
    oof[, idx] <- round_to_classes(oof[, idx], class_labels)
  }
}

meta <- train_meta_model_classification(oof, y, meta_params = cfg$meta_params)

# Evaluate training metrics
train_preds <- meta$predict(oof)
acc <- accuracy(y, train_preds)
f1s <- f1_scores(y, train_preds)
cat(sprintf("Training OOF Accuracy: %.4f\n", acc))
cat(sprintf("Training OOF Macro F1: %.4f\n", f1s$f1_macro))
cat(sprintf("Training OOF Weighted F1: %.4f\n", f1s$f1_weighted))

# Optionally evaluate on validation set
if (!is.null(cfg$val_size) && cfg$val_size > 0) {
  split <- split_train_val(X, y, val_size = cfg$val_size, random_state = cfg$random_state, stratify = TRUE)
  X_train <- split$train$X
  y_train <- split$train$y
  X_val <- split$val$X
  y_val <- split$val$y
  folds_val <- create_cv_splits(X_train, y_train, n_splits = cfg$n_splits, random_state = cfg$random_state, stratify = TRUE)
  res_val <- train_base_models_classification(
    X_train, y_train,
    model_names = cfg$base_models,
    model_params = cfg$model_params,
    folds = folds_val
  )
  oof_train <- res_val$oof
  base_models_val <- res_val$fitted_models
  # Round ordinal reg columns
  if (length(ordinal_indices) > 0) {
    for (idx in ordinal_indices) {
      oof_train[, idx] <- round_to_classes(oof_train[, idx], class_labels)
    }
  }
  meta_val <- train_meta_model_classification(oof_train, y_train, meta_params = cfg$meta_params)
  # Predict on validation set
  base_preds_val <- sapply(cfg$base_models, function(name) {
    mdl <- base_models_val[[name]]
    preds <- mdl$predict(X_val)
    if (name == "ordinal_reg") {
      preds <- round_to_classes(preds, class_labels)
    }
    return(preds)
  })
  meta_preds_val <- meta_val$predict(base_preds_val)
  acc_val <- accuracy(y_val, meta_preds_val)
  f1s_val <- f1_scores(y_val, meta_preds_val)
  cat(sprintf("Validation Accuracy: %.4f\n", acc_val))
  cat(sprintf("Validation Macro F1: %.4f\n", f1s_val$f1_macro))
  cat(sprintf("Validation Weighted F1: %.4f\n", f1s_val$f1_weighted))
}

# Save models and OOF predictions
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
saveRDS(base_models, file = file.path(output_dir, "base_models_classification.rds"))
saveRDS(meta, file = file.path(output_dir, "meta_classification.rds"))
saveRDS(oof, file = file.path(output_dir, "oof_classification.rds"))

cat("Finished training classification models.\n")