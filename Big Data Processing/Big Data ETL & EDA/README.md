
# Big Data ETL & EDA Project

This project demonstrates an end-to-end **ETL (Extract, Transform, Load)** and **Exploratory Data Analysis (EDA)** pipeline on a transactions dataset.

## 🚀 Quickstart
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. Run pipeline on **sample data**:
   ```bash
   make run-all
   ```

3. Fraud rate, processed data, and figures will be output to `data/processed/` and `reports/figures/`.

## 📂 Data
- `data/raw/samples/transactions_sample.csv`: a small synthetic dataset for demo.
- For full runs, place `transactions.csv` ans `merchant.csv` into `data/raw/` (ignored by git).

## 🛠️ Commands
- `make run-all`: Run ingest → transform → queries on sample data.
- `make download-data`: Download or place full `transactions.csv`.

## 📦 Repo Structure
```
src/portfolio_etl/    # ETL Python package
data/raw/samples/     # Small sample dataset
data/processed/       # Processed outputs (gitignored)
reports/figures/      # EDA figures (gitignored)
tests/                # Unit tests
```
