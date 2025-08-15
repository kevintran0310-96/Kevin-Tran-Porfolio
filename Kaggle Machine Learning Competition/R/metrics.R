## Metric functions for regression and classification tasks.

#' Compute root mean squared error (RMSE)
#'
#' @param y_true Numeric vector of true values
#' @param y_pred Numeric vector of predicted values
#' @return RMSE as numeric
rmse <- function(y_true, y_pred) {
  sqrt(mean((y_true - y_pred) ^ 2))
}

#' Compute accuracy
#'
#' @param y_true Integer or factor vector of true labels
#' @param y_pred Integer or factor vector of predicted labels
#' @return Accuracy as numeric
accuracy <- function(y_true, y_pred) {
  mean(y_true == y_pred)
}

#' Compute F1 score for a single class
f1_binary <- function(y_true, y_pred) {
  tp <- sum(y_true == 1 & y_pred == 1)
  fp <- sum(y_true == 0 & y_pred == 1)
  fn <- sum(y_true == 1 & y_pred == 0)
  if ((tp + fp) == 0 || (tp + fn) == 0) return(0)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  if (precision + recall == 0) return(0)
  2 * precision * recall / (precision + recall)
}

#' Compute macro and weighted F1 scores for multi‑class classification
#'
#' @param y_true Integer or factor vector of true labels
#' @param y_pred Integer or factor vector of predicted labels
#' @return A list with `f1_macro` and `f1_weighted` entries
f1_scores <- function(y_true, y_pred) {
  classes <- sort(unique(y_true))
  f1_vals <- numeric(length(classes))
  weights <- numeric(length(classes))
  for (i in seq_along(classes)) {
    cls <- classes[i]
    # For the one‑vs‑rest calculation, recode labels as 1 (positive) and 0 (negative)
    binary_true <- ifelse(y_true == cls, 1L, 0L)
    binary_pred <- ifelse(y_pred == cls, 1L, 0L)
    f1_vals[i] <- f1_binary(binary_true, binary_pred)
    weights[i] <- sum(y_true == cls) / length(y_true)
  }
  f1_macro <- mean(f1_vals, na.rm = TRUE)
  f1_weighted <- sum(f1_vals * weights)
  list(f1_macro = f1_macro, f1_weighted = f1_weighted)
}

#' Compute a confusion matrix
#'
#' @param y_true Integer or factor vector of true labels
#' @param y_pred Integer or factor vector of predicted labels
#' @return A matrix representing the confusion matrix
confusion_matrix <- function(y_true, y_pred) {
  tab <- table(true = factor(y_true, levels = sort(unique(y_true))),
               pred = factor(y_pred, levels = sort(unique(y_true))))
  return(as.matrix(tab))
}