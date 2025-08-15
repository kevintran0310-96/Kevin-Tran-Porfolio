library(testthat)
source("../R/data_prep.R")

test_that("preprocess_data returns correct dimensions and no NAs", {
  df <- data.frame(
    gender = c("Male", "Female", "Other", NA),
    whatIsYourHeightExpressItAsANumberInMetresM = c("1.70 - 1.80", "1.60", NA, "1.55 - 1.60"),
    feature1 = c(1, 2, 3, 4),
    target = c(10.0, 9.5, 8.0, 8.5)
  )
  res <- preprocess_data(df, target_col = "target", task = "regression")
  X <- res$X
  y <- res$y
  expect_equal(nrow(X), 4)
  expect_true(!any(is.na(X)))
  expect_equal(length(y), 4)
})

test_that("split_train_val stratifies classification", {
  X <- data.frame(f1 = 1:10)
  y <- c(rep(0L, 8), rep(1L, 2))
  res <- split_train_val(X, y, val_size = 0.2, random_state = 1, stratify = TRUE)
  expect_equal(length(res$val$y), 2)
  # Both classes should appear in validation set
  expect_equal(sort(unique(res$val$y)), sort(unique(y)))
})

test_that("create_cv_splits returns correct number of folds", {
  X <- data.frame(f1 = rnorm(30))
  y <- sample(c(0L, 1L), 30, replace = TRUE)
  folds <- create_cv_splits(X, y, n_splits = 5, random_state = 1, stratify = TRUE)
  expect_equal(length(folds), 5)
})