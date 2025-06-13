# Data Wrangling and Preprocessing for Food Delivery & Suburban Data

This project encompasses a comprehensive data wrangling and preprocessing pipeline. It demonstrates robust techniques for data cleaning, imputation of missing values, outlier detection, and preparing diverse datasets for further analysis and modeling.

The project is divided into two main tasks:

* **Task 1: Food Delivery Data Wrangling** - Focuses on cleaning and preparing a transactional food delivery dataset.

* **Task 2: Suburban Data Preprocessing & EDA** - Involves cleaning, exploring, transforming, and normalizing a suburban information dataset.

## Table of Contents

* [Project Overview](#project-overview)

* [Task 1: Food Delivery Data Wrangling](#task-1-food-delivery-data-wrangling)

    * [Problem Statement](#problem-statement)

    * [Solution Approach](#solution-approach)

        * [Error Detection and Fixing (\`dirty_data.csv\`)](#error-detection-and-fixing-dirty_data.csv)

        * [Missing Value Imputation (\`missing_data.csv\`)](#missing-value-imputation-missing_data.csv)

        * [Outlier Detection and Removal (\`outlier_data.csv\`)](#outlier-detection-and-removal-outlier_data.csv)

* [Task 2: Suburban Data Preprocessing & EDA](#task-2-suburban-data-preprocessing--eda)

    * [Dataset Overview](#dataset-overview)

    * [Preprocessing Steps](#preprocessing-steps)

    * [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)

    * [Transformation & Normalization](#transformation--normalization)

* [Technologies Used](#technologies-used)

* [Installation & Usage](#installation--usage)

* [Team Members](#team-members)

## Project Overview

The primary goal of this project is to transform raw, messy datasets into clean, reliable, and analysis-ready formats. This involves a series of systematic steps from identifying inconsistencies and errors to applying advanced techniques for data imputation and normalization, ensuring data quality and integrity.

## Task 1: Food Delivery Data Wrangling

### Problem Statement

Clean and preprocess a set of food delivery data from a restaurant in Melbourne, Australia. The dataset includes information on orders from three branches, each operating differently despite sharing the same menu. The project involves detecting and fixing errors, imputing missing values, and identifying and removing outliers across several datasets (\`dirty_data.csv\`, \`missing_data.csv\`, \`outlier_data.csv\`).

### Solution Approach

The solution involves a detailed process of data cleaning, imputation, and outlier handling, leveraging Python's data manipulation and graph theory libraries.

#### Error Detection and Fixing (\`dirty_data.csv\`)

This phase focused on identifying and correcting various inconsistencies and errors in the \`dirty_data.csv\` file:

* **Initial Data Understanding**: Loaded \`Group111_dirty_data.csv\`, \`branches.csv\`, \`edges.csv\`, and \`nodes.csv\`. Performed initial checks for duplicates, data types, and summary statistics. Identified anomalies in \`customer_lat\`, \`customer_lon\`, and \`branch_code\`.

* **Order Type and Time Mismatches**: Detected and corrected 37 entries where \`order_type\` (Breakfast, Lunch, Dinner) did not align with the \`time\` of order, based on predefined meal periods.

* **Inconsistent Branch Codes**: Standardized branch codes by converting lowercase entries (e.g., 'bk', 'tp', 'ns') to their correct uppercase counterparts (e.g., 'BK', 'TP', 'NS') using patterns derived from \`order_id\` prefixes. A total of 29 entries were corrected.

* **Customer Latitude and Longitude Anomalies**:

    * **Swapped Coordinates**: Identified and corrected cases where \`customer_lat\` and \`customer_lon\` values were mistakenly swapped by cross-referencing with valid geographical nodes. 4 entries were fixed.

    * **Incorrect Signs**: Addressed latitude and longitude values with incorrect positive/negative signs by flipping them to align with Melbourne's geographical range. This involved mapping to the nearest valid nodes.

* **Date Type Errors**: Corrected \`date\` column entries that were in invalid formats (e.g., 'YYYY-DD-MM') to the standard 'YYYY-MM-DD' format. This was verified by checking for valid month ranges.

* **Order Items Discrepancies**: Developed a function to identify and correct invalid food items within \`order_items\` lists, ensuring they align with the \`order_type\` (Breakfast, Lunch, Dinner menus). Invalid items were replaced with the closest valid item that maintained the original \`order_price\`. 37 orders were corrected.

* **Order Price Validation**: Recalculated \`order_price\` based on \`order_items\` and predefined item prices. Entries where the original \`order_price\` did not match the calculated price were corrected. 37 invalid order prices were fixed.

* **Distance to Customer KM Validation**: Implemented Dijkstra's algorithm using \`networkx\` to calculate the shortest geographical distance between branches and customers based on their coordinates and a network of nodes/edges. This validated and corrected \`distance_to_customer_KM\` values. 45 invalid distances were found and corrected.

* **Customer Loyalty Inconsistencies**: Identified and corrected records where \`customerHasloyalty?\` was 'Yes' but \`loyalty_start_date\` was missing or invalid. These were set to 'No Loyalty'.

* **Data Consolidation**: Throughout the cleaning process, a \`consolidated_changes\` DataFrame was maintained to log all corrections made, providing a clear audit trail of data transformations.

#### Missing Value Imputation (\`missing_data.csv\`)

This section addressed missing \`delivery_fee\` values in \`missing_data.csv\` using a linear regression model.

* **Preprocessing**: Prepared the dataset by converting \`order_type\` and \`branch_code\` into numerical representations suitable for modeling.

* **Linear Regression Model Development**: Built a linear regression model for each \`branch_code\` to predict \`delivery_fee\` based on \`distance_to_customer_KM\`, \`order_price\`, and \`customerHasloyalty?\`. The models were fine-tuned using transformations (e.g., \`boxcox\`) and best-seed evaluations to enhance prediction accuracy.

* **Imputation**: Applied the trained branch-specific models to impute missing \`delivery_fee\` values.

* **Validation**: Validated the imputed values using statistical analyses and visualizations to ensure robustness.

#### Outlier Detection and Removal (\`outlier_data.csv\`)

This phase focused on identifying and removing outlier rows in \`outlier_data.csv\`.

* **Data Loading and Understanding**: Loaded the \`outlier_data.csv\` and examined its structure.

* **Outlier Detection**: Identified outliers by comparing actual \`delivery_fee\` values with model-predicted values. Significant deviations were flagged as outliers.

* **Outlier Removal**: Removed the identified outlier rows from the dataset.

* **Validation**: Confirmed the removal of outliers and the integrity of the remaining data through statistical checks.

## Task 2: Suburban Data Preprocessing & EDA

### Dataset Overview

This task focuses on a \`suburb_info.xlsx\` dataset containing various features about Melbourne suburbs, including \`number_of_houses\`, \`number_of_units\`, \`median_income\`, \`median_house_price\`, and \`population\`.

### Preprocessing Steps

* **Data Type Conversion**: Converted \`aus_born_perc\`, \`median_income\`, and \`median_house_price\` from string/object types to numerical (float) by removing special characters (%, $, commas).

* **Handling Special Characters**: Cleaned numerical columns by stripping '%' signs, '$' signs, and commas.

### Exploratory Data Analysis (EDA)

* **Descriptive Statistics**: Generated summary statistics to understand central tendency, dispersion, and shape of the data distributions.

* **Distribution Analysis**: Visualized the distribution of key variables (e.g., histograms, density plots) to understand their spread and skewness.

* **Skewness Analysis**: Quantified the skewness of numerical variables to identify if transformations were necessary to achieve a more normal distribution.

* **Correlation Analysis**: Examined linear relationships between predictor variables and the target variable (\`median_house_price\`) using correlation matrices and heatmaps.

### Transformation & Normalization

Applied various transformation and normalization techniques to improve data distribution and scale for potential machine learning models:

* **Transformation**: Utilized Box-Cox and Yeo-Johnson transformations to reduce skewness and achieve more normal distributions for highly skewed features.

* **Normalization**: Employed RobustScaler (for handling outliers), QuantileTransformer (for non-linear transformations and uniform/normal distribution), and Normalizer (for scaling vectors to unit norm) to standardize feature scales.

## Technologies Used

* **Python**: Primary programming language.

* **Pandas**: For data manipulation and analysis.

* **NumPy**: For numerical operations.

* **Scipy**: For scientific computing and statistical functions.

* **Matplotlib & Seaborn**: For data visualization.

* **Scikit-learn**: For machine learning algorithms, preprocessing, and model selection utilities.

* **NetworkX**: For graph operations and shortest path calculations (Dijkstra's algorithm).

* **Jupyter Notebook**: For interactive development and documentation.

## Installation & Usage

To run this project, you will need Python and Jupyter Notebook installed.

**1. Clone the repository**

**2. Install the required Python packages:**
It's recommended to use a virtual environment.

```
pip install pandas numpy scipy matplotlib seaborn scikit-learn networkx
```

**3. Place the datasets:**
Ensure the following CSV and Excel files are in the same directory as the Jupyter notebooks:

* `Group111_dirty_data.csv`

* `missing_data.csv`

* `outlier_data.csv`

* `branches.csv`

* `edges.csv`

* `nodes.csv`

* `suburb_info.xlsx`

**4. Open and run the Jupyter Notebooks:**

```
jupyter notebook "A2_Task1.ipynb"
jupyter notebook "A2_task2.ipynb"
```

Execute the cells sequentially within each notebook to see the full data wrangling and preprocessing pipeline in action.
