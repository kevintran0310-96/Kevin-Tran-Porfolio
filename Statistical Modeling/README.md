# Kaggle Competition Solutions: Happiness Prediction (Regression) & Mental Health Classification

This repository showcases my solutions for two online Kaggle competitions, demonstrating my proficiency in machine learning, data preprocessing, model selection, and ensembling techniques.

## Project Overview

This project focuses on building robust predictive models for two distinct tasks based on a life and well-being survey dataset:

1. **Regression**: Predicting an individual's happiness level (continuous score).

2. **Classification**: Classifying individuals into categories of "perfect mental health" (5-class categorical score).

My solutions leverage various machine learning algorithms and preprocessing techniques to achieve strong performance in both competitions.

## Competition Details

### Regression Competition: Predict Happiness Level

* **Kaggle Link**: [Regression Contest 24 S1: Predict Happiness Level](https://www.kaggle.com/competitions/regression-contest-24-s-1-predict-happiness-level)

* **Goal**: Develop a regression model to predict the 'happiness' variable based on survey responses.

* **My Rank**: 1st out of 230 participants.

* **Evaluation Metric**: Root Mean Squared Error (rMSE).

### Classification Competition: Predict Perfect Mental Health

* **Kaggle Link**: [Classification 24.1: Predict Perfect Mental Health](https://www.kaggle.com/competitions/classification-24-1-predict-perfect-mental-health)

* **Goal**: Develop a 5-class classification model to predict the 'perfectMentalHealth' variable based on survey responses.

* **My Rank**: 61st out of 225 participants.

* **Evaluation Metric**: F1-Score.

## Solution Approach

### Regression Model

For the regression task, my approach involved a multi-model ensemble strategy.

1. **Data Loading and Preprocessing**:

   * Loaded `regression_train.csv` and `regression_test.csv`.

   * Handled missing values using median imputation.

   * Ensured all relevant columns were numeric.

2. **Individual Model Training**:

   * **Random Forest (rf)**: Trained with `mtry = 12` and `ntree = 500`.

   * **Support Vector Machine (svmRadial)**: Trained with `C = 5` and `sigma = 0.005`.

   * **XGBoost (xgbTree)**: Tuned with `nrounds = 500`, `max_depth = 3`, `eta = 0.05`, `gamma = 0.5`, `colsample_bytree = 0.5`, `min_child_weight = 5`, `subsample = 0.9`.

   * All models used 5-fold cross-validation (`method = "cv", number = 5`).

3. **Ensembling**:

   * The predictions from the trained XGBoost, Random Forest, and SVM models were combined into a new training dataset.

   * A final linear regression model (`method = "lm"`) was trained on these combined predictions to produce the ultimate happiness prediction. This meta-model learned the optimal weighting of each base model's output.

### Classification Model

For the classification task, I focused on feature engineering and an optimized XGBoost model.

1. **Data Loading and Encoding**:

   * Loaded `classification_train.csv` and `classification_test.csv`.

   * **Ordinal Encoding**: Applied to `income`, `whatIsYourHeightExpressItAsANumberInMetresM`, `howDoYouReconcileSpiritualBeliefsWithScientificOrRationalThinki`, `howOftenDoYouFeelSociallyConnectedWithYourPeersAndFriends`, `doYouHaveASupportSystemOfFriendsAndFamilyToTurnToWhenNeeded`, `howOftenDoYouParticipateInSocialActivitiesIncludingClubsSportsV`, and `doYouFeelComfortableEngagingInConversationsWithPeopleFromDiffer`.

   * **Binary Encoding**: Applied to `doYouFeelASenseOfPurposeAndMeaningInYourLife104`, `doYouFeelASenseOfPurposeAndMeaningInYourLife105`, and `gender`.

   * Missing values were handled by replacing them with `-999`.

2. **Feature Selection**:

   * After initial experimentation, Recursive Feature Elimination (RFE) was used to identify a subset of top features, including: `alwaysLoveAndCareForYourself`, `lifeIsGood`, `alwaysEngageInPreparingAndUsingYourSkillsAndTalentsInOrderToGai`, `alwaysCalm`, and `alwaysStressed`.

   * The datasets were then subsetted to include only these selected features.

3. **XGBoost Model**:

   * The target variable `perfectMentalHealth` was converted to a 0-based index for XGBoost.

   * The XGBoost model was trained with specific parameters for multi-class classification: `objective = "multi:softmax"`, `eval_metric = "merror"`, `num_class` (set dynamically), `max_depth = 5`, `min_child_weight = 0.8`, `gamma = 0.1`, `subsample = 0.7`, `colsample_bytree = 0.6`, `eta = 0.01`, and `nrounds = 1000`.

## Results & Performance

* **Regression Competition**: Achieved 1st place, demonstrating highly accurate predictions for happiness levels. The ensemble approach effectively minimized the rMSE.

* **Classification Competition**: Achieved 61st place, indicating a strong performance in classifying mental health categories. The strategic feature engineering combined with a robust XGBoost model proved effective.

## Repository Structure

```bash
├── Prediction and Classification Models.ipynb  # Jupyter Notebook containing all R code for both competitions
├── regression_train.csv                       # Training data for regression
├── regression_test.csv                        # Test data for regression
├── classification_train.csv                   # Training data for classification
├── classification_test.csv                    # Test data for classification
└── README.md                                  # This README file
```

## Installation & Usage

The project code is primarily written in R and is contained within the `Prediction and Classification Models.ipynb` Jupyter Notebook.

To run the code locally, you will need:

* R installed on your system.

* Jupyter Notebook (or RStudio with Jupyter kernel support).

**1. Clone the repository:**

**2. Open the Jupyter Notebook:**

```bash
jupyter notebook "Prediction and Classification Models.ipynb"
```

**3. Install R Packages (if not already installed):**
The notebook includes a function `install_if_missing` to handle package installations automatically. Ensure you run these cells first. The main packages used are:

* `caret`

* `xgboost`

* `randomForest`

* `e1071`

* `readr`

* `dplyr`

**4. Run the notebook cells sequentially.**
The notebook is structured to demonstrate each step of the analysis, from data loading and preprocessing to model training and prediction submission.

## Contact

Feel free to reach out if you have any questions or would like to discuss this project further.

**Name**: Quoc Khoa Tran

**Kaggle Name/ID**: quockhoatran

**Email**: khoatran031096@gmail.com


