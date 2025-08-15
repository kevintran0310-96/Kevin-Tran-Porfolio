# Data Wrangling Project

This repository contains a refactored and production‑ready implementation of a
data wrangling pipeline originally developed as part of a university course.
The goal of this project is to demonstrate industry‑standard practices for
cleaning and preprocessing messy datasets as a Data Scientist or AI engineer.

The original assignment consisted of three subtasks performed on three
separate datasets:

1. **Dirty data** – detect and fix invalid values such as inconsistent
   branch codes, anomalous latitude/longitude, mismatches between the
   declared meal type and the time of day, incorrect order item counts and
   prices, and erroneous distances.
2. **Missing data** – build a predictive model to impute missing delivery
   fees based on explanatory variables (e.g. weekend indicator, time of
   day, distance to customer, customer loyalty status).
3. **Outlier data** – identify and remove outliers using a trained
   regression model, then validate the cleaned dataset.

This repository reorganises the above tasks into a modular Python package
with clear separation of concerns. It also includes unit tests, a simple
configuration system, type hints, docstrings and logging.  By following
the steps below you can reproduce the analysis and demonstrate your
engineering skills to prospective employers.

## Project structure

```
data_wrangle_project/
├── README.md                ← high‑level overview and instructions
├── requirements.txt         ← Python dependencies
├── configs/
│   └── default.yaml         ← configuration file (e.g. file paths)
├── src/
│   └── datawrangle/
│       ├── __init__.py      ← exposes public API
│       ├── cleaning.py      ← core cleaning functions
│       ├── io.py            ← I/O helpers for reading/writing data
│       └── validation.py    ← data schema definitions
├── tests/
│   └── test_cleaning.py     ← unit tests for cleaning functions
└── .github/
    └── workflows/
        └── ci.yml           ← minimal CI configuration running linting & tests
```

## Getting started

1. **Install dependencies** – Create a virtual environment and install the
   required packages:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Edit configuration** – Adjust `configs/default.yaml` to point to your
   own input CSV files and output locations.  The configuration file is
   written in YAML and loaded by the I/O helpers.

3. **Run cleaning pipeline** – Use the functions exposed in
   `src/datawrangle/cleaning.py` to load, validate and clean your data.
   For example:

   ```python
   from datawrangle.io import load_datasets
   from datawrangle.cleaning import (correct_branch_code,
                                     correct_lat_lon,
                                     correct_order_items,
                                     compute_time_features)

   datasets = load_datasets()
   dirty = datasets['dirty']
   dirty = correct_branch_code(dirty)
   dirty = correct_lat_lon(dirty)
   dirty = correct_order_items(dirty)
   dirty = compute_time_features(dirty)
   # … further cleaning and modelling
   ```

4. **Run tests** – Execute the unit tests with pytest to ensure your
   changes don’t break existing functionality:

   ```bash
   pytest
   ```

## Why this refactor?

The original notebooks and scripts contained exploration, plots and long
procedural code mixed together.  While that style is acceptable for a
class assignment, it isn’t well suited for real‑world projects.  This
refactor introduces the following improvements:

* **Separation of concerns** – logic is split into small functions under
  `src/datawrangle` rather than thousands of lines in a single script.
* **Reusability** – functions accept a DataFrame and return a new
  DataFrame, enabling chaining and reuse in other projects.
* **Documentation** – every public function has a docstring and type
  hints, making it easier for others to understand how to use it.
* **Logging** – printing directly to stdout has been replaced with the
  Python logging module.  You can control verbosity via standard logging
  configuration.
* **Configuration** – file paths and other parameters are externalised
  into YAML instead of being hard coded.
* **Validation** – basic data contracts are provided via pandera to
  validate input and output DataFrames.
* **Testing** – unit tests ensure that your cleaning functions behave as
  expected and make it easier to extend or refactor code without fear.

By following these practices you signal to employers that you can write
maintainable, production‑ready code rather than ad‑hoc scripts.
