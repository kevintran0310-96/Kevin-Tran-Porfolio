## Convenience script to install all R package dependencies for the project.

packages <- c(
  "yaml",
  "randomForest",
  "e1071",
  "xgboost",
  "nnet",
  "caret",
  "MLmetrics",
  "dplyr",
  "data.table",
  "testthat"
)

installed <- rownames(installed.packages())
to_install <- setdiff(packages, installed)

if (length(to_install) > 0) {
  message("Installing packages: ", paste(to_install, collapse = ", "))
  install.packages(to_install, repos = "https://cloud.r-project.org", dependencies = TRUE)
} else {
  message("All required packages are already installed.")
}