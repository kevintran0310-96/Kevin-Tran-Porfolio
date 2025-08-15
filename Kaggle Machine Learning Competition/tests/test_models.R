library(testthat)
source("../R/data_prep.R")
source("../R/models.R")
source("../R/train_utils.R")
source("../R/metrics.R")

test_that("train_base_models_regression returns correct oof shape", {
  set.seed(123)
  X <- data.frame(a = rnorm(40), b = runif(40))
  y <- rnorm(40)
  folds <- create_cv_splits(X, y, n_splits = 4, random_state = 1, stratify = FALSE)
  res <- train_base_models_regression(X, y, model_names = c("rf", "svm"), model_params = list(rf = list(ntree = 10), svm = list(cost = 0.5, type = "eps-regression")), folds = folds)
  expect_equal(dim(res$oof), c(40, 2))
  # meta model
  meta <- train_meta_model_regression(res$oof, y, meta_params = list())
  preds <- meta$predict(res$oof)
  expect_equal(length(preds), 40)
})

test_that("train_base_models_classification returns correct oof shape", {
  set.seed(123)
  X <- data.frame(a = rnorm(45), b = runif(45))
  y <- sample(c(-2L, -1L, 0L, 1L, 2L), 45, replace = TRUE)
  folds <- create_cv_splits(X, y, n_splits = 3, random_state = 1, stratify = TRUE)
  res <- train_base_models_classification(X, y, model_names = c("rf", "svm"), model_params = list(rf = list(ntree = 10), svm = list(cost = 0.5)), folds = folds)
  expect_equal(dim(res$oof), c(45, 2))
  # meta model classification
  meta <- train_meta_model_classification(res$oof, y, meta_params = list(maxit = 50))
  preds <- meta$predict(res$oof)
  expect_equal(length(preds), 45)
})