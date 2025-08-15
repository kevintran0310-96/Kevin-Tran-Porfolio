#!/usr/bin/env Rscript

## Script to train regression models for the happiness prediction task.
##
## Usage:
##   Rscript scripts/run_regression.R --config configs/regression.yaml --output_dir data/processed

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

# Set seed
set.seed(cfg$random_state)

# Load data and preprocess
raw_df <- load_dataset(cfg$train_path)
prep <- preprocess_data(raw_df, target_col = cfg$target_column, task = "regression")
X <- prep$X
y <- prep$y

# Create cross‑validation folds
folds <- create_cv_splits(X, y, n_splits = cfg$n_splits, random_state = cfg$random_state, stratify = TRUE)

# Train base models and meta model
train_res <- train_base_models_regression(
  X = X,
  y = y,
  model_names = cfg$base_models,
  model_params = cfg$model_params,
  folds = folds
)
oof <- train_res$oof
base_models <- train_res$fitted_models
meta <- train_meta_model_regression(oof, y, meta_params = cfg$meta_params)

# Evaluate training performance on OOF predictions
train_preds <- meta$predict(oof)
train_rmse <- rmse(y, train_preds)
cat(sprintf("Training OOF RMSE: %.4f\n", train_rmse))

# Optionally evaluate on a hold‑out validation set
if (!is.null(cfg$val_size) && cfg$val_size > 0) {
  split <- split_train_val(X, y, val_size = cfg$val_size, random_state = cfg$random_state, stratify = TRUE)
  X_train <- split$train$X
  y_train <- split$train$y
  X_val <- split$val$X
  y_val <- split$val$y
  # Refit base models on training partition
  folds_val <- create_cv_splits(X_train, y_train, n_splits = cfg$n_splits, random_state = cfg$random_state, stratify = TRUE)
  res_val <- train_base_models_regression(
    X_train, y_train,
    model_names = cfg$base_models,
    model_params = cfg$model_params,
    folds = folds_val
  )
  oof_train <- res_val$oof
  base_models_val <- res_val$fitted_models
  meta_val <- train_meta_model_regression(oof_train, y_train, meta_params = cfg$meta_params)
  # Predict on validation set
  base_preds_val <- sapply(cfg$base_models, function(name) {
    mdl <- base_models_val[[name]]
    mdl$predict(X_val)
  })
  meta_preds_val <- meta_val$predict(base_preds_val)
  val_rmse <- rmse(y_val, meta_preds_val)
  cat(sprintf("Validation RMSE: %.4f\n", val_rmse))
}

# Save models and OOF predictions
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
saveRDS(base_models, file = file.path(output_dir, "base_models_regression.rds"))
saveRDS(meta, file = file.path(output_dir, "meta_regression.rds"))
saveRDS(oof, file = file.path(output_dir, "oof_regression.rds"))

cat("Finished training regression models.\n")