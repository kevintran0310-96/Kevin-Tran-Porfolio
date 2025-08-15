## Data loading, preprocessing and splitting utilities for the mental health project.

#' Load a CSV file as a data.frame
#'
#' @param file_path Character. Path to the CSV file.
#' @return A data.frame containing the raw data.
load_dataset <- function(file_path) {
  df <- utils::read.csv(file_path, stringsAsFactors = FALSE)
  return(df)
}

#' Parse a height value expressed as a number or a range.
#'
#' Some survey responses encode height as a range such as "165 - 170".  This
#' helper function returns the mid‑point of the range as a numeric value.  If the
#' value is a simple numeric string it is converted directly.  Non‑parseable
#' values return NA.
#'
#' @param value A scalar value (character or numeric).
#' @return Numeric mid‑point or NA.
parse_height <- function(value) {
  if (is.na(value) || value == "") {
    return(NA_real_)
  }
  s <- trimws(as.character(value))
  # Try numeric conversion
  suppressWarnings(num <- as.numeric(s))
  if (!is.na(num)) {
    return(num)
  }
  # Try pattern like "165-170" or "1.65 - 1.70"
  m <- regmatches(s, regexec("^([0-9]+\.?[0-9]*)\s*[-–]\s*([0-9]+\.?[0-9]*)$", s))[[1]]
  if (length(m) == 3) {
    low <- as.numeric(m[2])
    high <- as.numeric(m[3])
    return((low + high) / 2)
  }
  return(NA_real_)
}

#' Compute the statistical mode of a vector.
#'
#' @param x A vector.
#' @return The most common value in `x`.  In case of ties the first mode is returned.
stat_mode <- function(x) {
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

#' Preprocess the dataset by cleaning, encoding and imputing missing values.
#'
#' This function implements the following steps:
#'   * Remove the target column and return it separately as `y`.
#'   * Map gender strings to integers.
#'   * Parse height ranges into numeric mid‑points.
#'   * Identify categorical columns (character) and convert them to integers via factor encoding.
#'   * Impute missing numeric values with the median and missing categorical values with the mode.
#'
#' @param df data.frame. The raw dataset.
#' @param target_col Character. Name of the target column.
#' @param task Character. Either "regression" or "classification".
#' @return A list with elements `X` (preprocessed features) and `y` (target).
preprocess_data <- function(df, target_col, task = c("regression", "classification")) {
  task <- match.arg(task)
  df <- as.data.frame(df)
  # Extract target
  y <- df[[target_col]]
  df[[target_col]] <- NULL
  if (task == "classification") {
    y <- as.integer(y)
  } else {
    y <- as.numeric(y)
  }
  # Map gender
  if ("gender" %in% names(df)) {
    df$gender <- tolower(df$gender)
    mapping <- c("male" = 0L, "female" = 1L)
    df$gender <- ifelse(df$gender %in% names(mapping), mapping[df$gender], -1L)
    df$gender <- as.integer(df$gender)
  }
  # Parse height
  if ("whatIsYourHeightExpressItAsANumberInMetresM" %in% names(df)) {
    df$height <- vapply(df$whatIsYourHeightExpressItAsANumberInMetresM, parse_height, numeric(1))
    df$whatIsYourHeightExpressItAsANumberInMetresM <- NULL
  }
  # Identify character columns
  char_cols <- names(df)[sapply(df, is.character)]
  # Convert characters to factor integers
  for (col in char_cols) {
    df[[col]][is.na(df[[col]])] <- NA_character_
    # Replace empty strings with NA
    df[[col]][df[[col]] == ""] <- NA_character_
    df[[col]] <- as.integer(factor(df[[col]], exclude = NULL))
  }
  # Impute missing values
  for (col in names(df)) {
    if (is.numeric(df[[col]])) {
      if (any(is.na(df[[col]]))) {
        med <- median(df[[col]], na.rm = TRUE)
        df[[col]][is.na(df[[col]])] <- med
      }
    } else {
      # Mode imputation for other types
      if (any(is.na(df[[col]]))) {
        mode_val <- stat_mode(df[[col]][!is.na(df[[col]])])
        df[[col]][is.na(df[[col]])] <- mode_val
      }
    }
  }
  return(list(X = df, y = y))
}

#' Split the data into training and validation sets.
#'
#' @param X data.frame of features
#' @param y vector of targets
#' @param val_size Numeric fraction of observations to allocate to the validation set
#' @param random_state Integer seed for reproducibility
#' @param stratify Logical. If TRUE and the task is classification, perform a stratified split.
#' @return A list containing `train` and `val` elements, each with `X` and `y` components
split_train_val <- function(X, y, val_size = 0.2, random_state = 42L, stratify = TRUE) {
  set.seed(random_state)
  n <- nrow(X)
  # Determine indices for validation
  if (stratify && length(unique(y)) > 1) {
    # Use caret to create stratified partition
    if (!requireNamespace("caret", quietly = TRUE)) {
      stop("Package 'caret' is required for stratified splitting. Please install it.")
    }
    if (is.factor(y) || is.character(y)) {
      strat_y <- y
    } else {
      # For regression, bin y into quantiles
      q <- min(5, floor(length(y) / 10))
      strat_y <- cut(y, breaks = unique(stats::quantile(y, probs = seq(0, 1, length.out = q + 1), na.rm = TRUE)), include.lowest = TRUE)
    }
    in_val <- caret::createDataPartition(strat_y, p = val_size, list = FALSE)
  } else {
    # Simple random sample
    in_val <- sample(seq_len(n), size = floor(val_size * n))
  }
  val_idx <- sort(in_val)
  train_idx <- setdiff(seq_len(n), val_idx)
  list(
    train = list(X = X[train_idx, , drop = FALSE], y = y[train_idx]),
    val   = list(X = X[val_idx, , drop = FALSE], y = y[val_idx])
  )
}

#' Generate k‑fold cross‑validation indices.
#'
#' @param X data.frame of features
#' @param y vector of targets
#' @param n_splits Number of folds
#' @param random_state Seed for reproducibility
#' @param stratify Logical. If TRUE, perform stratified folds (classification) or use quantile
#'                 binning (regression).
#' @return A list of length `n_splits` with `train_idx` and `val_idx` integer vectors.
create_cv_splits <- function(X, y, n_splits = 5L, random_state = 42L, stratify = TRUE) {
  set.seed(random_state)
  folds <- vector("list", n_splits)
  n <- nrow(X)
  if (stratify && length(unique(y)) > 1) {
    if (!requireNamespace("caret", quietly = TRUE)) {
      stop("Package 'caret' is required for stratified k‑fold splitting. Please install it.")
    }
    if (is.factor(y) || is.character(y)) {
      strat_y <- y
    } else {
      # Bin continuous target for regression
      q <- min(n_splits, floor(length(y) / 10))
      strat_y <- cut(y, breaks = unique(stats::quantile(y, probs = seq(0, 1, length.out = q + 1), na.rm = TRUE)), include.lowest = TRUE)
    }
    fold_indices <- caret::createFolds(strat_y, k = n_splits, list = TRUE, returnTrain = FALSE)
    for (i in seq_len(n_splits)) {
      val_idx <- fold_indices[[i]]
      train_idx <- setdiff(seq_len(n), val_idx)
      folds[[i]] <- list(train_idx = train_idx, val_idx = val_idx)
    }
  } else {
    # Non-stratified random KFold
    indices <- sample(seq_len(n))
    fold_sizes <- rep(floor(n / n_splits), n_splits)
    remainder <- n %% n_splits
    if (remainder > 0) {
      fold_sizes[seq_len(remainder)] <- fold_sizes[seq_len(remainder)] + 1
    }
    current <- 1L
    for (i in seq_len(n_splits)) {
      start <- current
      end <- current + fold_sizes[i] - 1
      val_idx <- indices[start:end]
      train_idx <- setdiff(seq_len(n), val_idx)
      folds[[i]] <- list(train_idx = train_idx, val_idx = val_idx)
      current <- end + 1
    }
  }
  return(folds)
}