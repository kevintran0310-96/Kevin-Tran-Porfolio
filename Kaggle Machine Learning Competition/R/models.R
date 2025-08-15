## Model definitions and training functions for the mental health project.

## Helper functions to instantiate and fit base learners for regression and classification.

#' Get a trained base learner
#'
#' @param name Character, one of "rf", "svm", "xgb", or "ordinal_reg" (classification only).
#' @param X data.frame of predictors
#' @param y vector of responses (numeric for regression, integer for classification)
#' @param params List of hyperparameters for the model
#' @param task Character, "regression" or "classification"
#' @return A list with elements `model` and `predict` (a function taking newdata)
train_base_learner <- function(name, X, y, params, task) {
  if (task == "regression") {
    if (name == "rf") {
      if (!requireNamespace("randomForest", quietly = TRUE)) {
        stop("Package 'randomForest' is required for rf. Please install it.")
      }
      mod <- randomForest::randomForest(x = X, y = y,
                                        ntree = params$ntree %||% 100,
                                        mtry = params$mtry %||% floor(sqrt(ncol(X))))
      predict_fun <- function(newdata) predict(mod, newdata = newdata)
      return(list(model = mod, predict = predict_fun))
    } else if (name == "svm") {
      if (!requireNamespace("e1071", quietly = TRUE)) {
        stop("Package 'e1071' is required for svm. Please install it.")
      }
      mod <- e1071::svm(x = X, y = y,
                        type = params$type %||% "eps-regression",
                        cost = params$cost %||% 1,
                        gamma = params$gamma %||% 0.1,
                        kernel = params$kernel %||% "radial")
      predict_fun <- function(newdata) predict(mod, newdata = newdata)
      return(list(model = mod, predict = predict_fun))
    } else if (name == "xgb") {
      if (!requireNamespace("xgboost", quietly = TRUE)) {
        stop("Package 'xgboost' is required for xgb. Please install it.")
      }
      # Convert to xgb.DMatrix
      dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y)
      param_list <- list(
        objective = "reg:squarederror",
        eta = params$eta %||% 0.05,
        max_depth = params$max_depth %||% 6,
        subsample = params$subsample %||% 0.8,
        colsample_bytree = params$colsample_bytree %||% 0.8,
        lambda = params$lambda %||% 1.0
      )
      nrounds <- params$nrounds %||% 400
      mod <- xgboost::xgb.train(
        params = param_list,
        data = dtrain,
        nrounds = nrounds,
        verbose = 0
      )
      predict_fun <- function(newdata) {
        preds <- xgboost::predict(mod, as.matrix(newdata))
        return(preds)
      }
      return(list(model = mod, predict = predict_fun))
    } else {
      stop(paste("Unknown regression model", name))
    }
  } else { # classification
    if (name == "rf") {
      if (!requireNamespace("randomForest", quietly = TRUE)) {
        stop("Package 'randomForest' is required for rf. Please install it.")
      }
      # Convert y to factor
      mod <- randomForest::randomForest(x = X, y = as.factor(y),
                                        ntree = params$ntree %||% 100,
                                        mtry = params$mtry %||% floor(sqrt(ncol(X))),
                                        classwt = params$classwt)
      predict_fun <- function(newdata) as.integer(predict(mod, newdata = newdata))
      return(list(model = mod, predict = predict_fun))
    } else if (name == "svm") {
      if (!requireNamespace("e1071", quietly = TRUE)) {
        stop("Package 'e1071' is required for svm. Please install it.")
      }
      mod <- e1071::svm(x = X, y = as.factor(y),
                        cost = params$cost %||% 1,
                        gamma = params$gamma %||% 0.1,
                        kernel = params$kernel %||% "radial",
                        probability = FALSE)
      predict_fun <- function(newdata) as.integer(predict(mod, newdata = newdata))
      return(list(model = mod, predict = predict_fun))
    } else if (name == "xgb") {
      if (!requireNamespace("xgboost", quietly = TRUE)) {
        stop("Package 'xgboost' is required for xgb. Please install it.")
      }
      num_class <- length(unique(y))
      dtrain <- xgboost::xgb.DMatrix(data = as.matrix(X), label = y)
      param_list <- list(
        objective = params$objective %||% "multi:softprob",
        num_class = num_class,
        eta = params$eta %||% 0.05,
        max_depth = params$max_depth %||% 6,
        subsample = params$subsample %||% 0.8,
        colsample_bytree = params$colsample_bytree %||% 0.8,
        eval_metric = params$eval_metric %||% "mlogloss"
      )
      nrounds <- params$nrounds %||% 400
      mod <- xgboost::xgb.train(
        params = param_list,
        data = dtrain,
        nrounds = nrounds,
        verbose = 0
      )
      predict_fun <- function(newdata) {
        # Predict probabilities and take argmax
        probs <- matrix(xgboost::predict(mod, as.matrix(newdata)), ncol = num_class, byrow = TRUE)
        preds <- max.col(probs, ties.method = "random")
        # XGBoost labels classes from 0 to num_class-1; adjust to original labels order
        # We assume y values are sorted; map predicted indices to sorted unique labels
        classes <- sort(unique(y))
        return(as.integer(classes[preds]))
      }
      return(list(model = mod, predict = predict_fun))
    } else if (name == "ordinal_reg") {
      # Use linear regression to predict a continuous score and round later
      mod <- stats::lm(y ~ ., data = cbind(y = y, X))
      predict_fun <- function(newdata) {
        as.numeric(stats::predict(mod, newdata))
      }
      return(list(model = mod, predict = predict_fun))
    } else {
      stop(paste("Unknown classification model", name))
    }
  }
}

# Provide `%||%` operator for default values
`%||%` <- function(x, y) if (!is.null(x)) x else y

#' Train base models for regression and produce out‑of‑fold predictions
#'
#' @param X data.frame of features
#' @param y numeric vector of responses
#' @param model_names character vector of base model names
#' @param model_params list of lists with hyperparameters for each model
#' @param folds list of folds with train_idx and val_idx
#' @return A list with `oof` matrix and `fitted_models` list
train_base_models_regression <- function(X, y, model_names, model_params, folds) {
  n <- nrow(X)
  m <- length(model_names)
  oof <- matrix(0, nrow = n, ncol = m)
  fitted_models <- list()
  names(fitted_models) <- model_names
  # Train models on each fold to generate OOF predictions
  for (j in seq_along(model_names)) {
    name <- model_names[j]
    params <- model_params[[name]] %||% list()
    oof_pred <- numeric(n)
    for (fold in folds) {
      train_idx <- fold$train_idx
      val_idx <- fold$val_idx
      fit <- train_base_learner(name, X[train_idx, , drop = FALSE], y[train_idx], params, task = "regression")
      preds <- fit$predict(X[val_idx, , drop = FALSE])
      oof_pred[val_idx] <- preds
    }
    oof[, j] <- oof_pred
    # Fit on full data
    full_fit <- train_base_learner(name, X, y, params, task = "regression")
    fitted_models[[name]] <- full_fit
  }
  return(list(oof = oof, fitted_models = fitted_models))
}

#' Train meta‑model for regression
train_meta_model_regression <- function(oof, y, meta_params) {
  # Use linear regression for meta model
  df <- as.data.frame(oof)
  colnames(df) <- paste0("m", seq_len(ncol(df)))
  df$y <- y
  formula <- stats::as.formula(paste("y ~", paste(colnames(df)[1:ncol(oof)], collapse = "+")))
  mod <- stats::lm(formula, data = df)
  predict_fun <- function(newdata) stats::predict(mod, newdata = as.data.frame(newdata))
  return(list(model = mod, predict = predict_fun))
}

#' Train base models for classification and produce out‑of‑fold predictions
#'
#' @param X data.frame of features
#' @param y integer vector of class labels
#' @param model_names character vector of base model names
#' @param model_params list of lists with hyperparameters for each model
#' @param folds list of folds with train_idx and val_idx
#' @return A list with `oof` matrix and `fitted_models` list
train_base_models_classification <- function(X, y, model_names, model_params, folds) {
  n <- nrow(X)
  m <- length(model_names)
  oof <- matrix(0, nrow = n, ncol = m)
  fitted_models <- list()
  names(fitted_models) <- model_names
  for (j in seq_along(model_names)) {
    name <- model_names[j]
    params <- model_params[[name]] %||% list()
    oof_pred <- numeric(n)
    for (fold in folds) {
      train_idx <- fold$train_idx
      val_idx <- fold$val_idx
      fit <- train_base_learner(name, X[train_idx, , drop = FALSE], y[train_idx], params, task = "classification")
      preds <- fit$predict(X[val_idx, , drop = FALSE])
      oof_pred[val_idx] <- preds
    }
    oof[, j] <- oof_pred
    full_fit <- train_base_learner(name, X, y, params, task = "classification")
    fitted_models[[name]] <- full_fit
  }
  return(list(oof = oof, fitted_models = fitted_models))
}

#' Train meta‑model for classification
train_meta_model_classification <- function(oof, y, meta_params) {
  if (!requireNamespace("nnet", quietly = TRUE)) {
    stop("Package 'nnet' is required for multinomial logistic regression. Please install it.")
  }
  df <- as.data.frame(oof)
  colnames(df) <- paste0("m", seq_len(ncol(df)))
  df$y <- as.factor(y)
  formula <- stats::as.formula(paste("y ~", paste(colnames(df)[1:ncol(oof)], collapse = "+")))
  mod <- nnet::multinom(formula, data = df, maxit = meta_params$maxit %||% 200, trace = FALSE)
  predict_fun <- function(newdata) {
    pred <- predict(mod, newdata = as.data.frame(newdata))
    # Convert factor back to integer labels
    as.integer(as.character(pred))
  }
  return(list(model = mod, predict = predict_fun))
}