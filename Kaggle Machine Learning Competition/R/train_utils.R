## Helper functions used by the training scripts

#' Round continuous predictions to the nearest class label
#'
#' @param preds Numeric vector of continuous predictions
#' @param class_labels Numeric vector of allowed labels (sorted)
#' @return Integer vector of rounded labels
round_to_classes <- function(preds, class_labels) {
  class_labels <- sort(unique(class_labels))
  sapply(preds, function(p) {
    class_labels[which.min(abs(class_labels - p))]
  })
}