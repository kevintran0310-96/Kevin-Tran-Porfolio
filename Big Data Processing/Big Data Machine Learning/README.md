# Big Data Fraud Detection — Spark (A2A)

This repository contains a **reproducible** implementation of a Big Data ETL + EDA + Modeling workflow for **e‑commerce fraud detection** using **PySpark** (Part A of the assignment). It follows a schema‑first, pipeline‑driven approach and includes sample data, configs, and scripts to make it easy to run locally.

> Context: The assignment asks you to build fraud detection with Spark MLlib/Streaming using customer & browsing behaviour, then evaluate RF/GBT and do K‑Means clustering for behaviour insight. fileciteturn1file0

## ✨ What’s inside
- **`src/`** — modular Python package with ETL, feature engineering, training, and clustering scripts
- **`configs/`** — YAML config (file paths, column names, hyperparams) and optional schema JSON
- **`data/sample/`** — small CSV slices for quick smoke‑tests (replace with full dataset under `data/raw/`)
- **`notebooks/`** — your original notebook (`A2A_34124888.ipynb`) for narrative EDA/plots (kept reproducible)
- **`docs/`** — spec PDF for reference (optional)
- **CI‑friendly** layout + `Makefile` targets

## 🧱 Project Structure
```
fraud-etl-eda-repo/
├─ src/
│  ├─ etl.py                # Load → validate schema → clean → persist parquet
│  ├─ features.py           # L1/L2/L3 event features, ratios, joins, label attach
│  ├─ train.py              # RF/GBT pipelines + metrics (AUC/PR, confusion)
│  ├─ cluster.py            # K-means with elbow/silhouette helpers
│  ├─ utils/
│  │  ├─ io.py              # Spark session & I/O helpers
│  │  └─ schemas.py         # (Optional) explicit StructTypes loaded from JSON
│  └─ __init__.py
├─ configs/
│  ├─ config.yaml           # paths, column mapping, model params
│  └─ schema.json           # (optional) explicit schema; leave null to infer
├─ data/
│  ├─ sample/               # tiny CSV slices for quick runs
│  └─ raw/                  # put full CSVs here (gitignored)
├─ notebooks/
│  └─ A2A_34124888.ipynb    # provided notebook
├─ docs/
│  └─ A2A_Specification.2024S2.pdf
├─ Makefile
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## 🚀 Quickstart
```bash
# 1) create and activate a venv (any tool is fine)
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate

# 2) install deps
pip install -r requirements.txt

# 3) smoke test on sample data
make etl        # writes parquet to data/processed/
make features   # builds features.parquet
make train      # trains RF/GBT on sample, prints metrics
make cluster    # runs K-means on behaviour features
```

> For **full evaluation**, place the complete CSV files in `data/raw/` and update `configs/config.yaml` if names differ.

## ⚙️ Configuration
All parameters live in **`configs/config.yaml`** (paths, column names, model params). The code reads this file at runtime so you don’t need to edit Python files to adjust settings.

Key mappings you may need to confirm:
- **Event types** → levels: `L1=[AP,ATC,CO]`, `L2=[VC,VP,VI,SER]`, `L3=[SCR,HP,CL]`
- **Joins** to attach customer, session, and fraud labels
- **Time‑of‑day** bucketing logic and ratio columns (L1/L2/%)

## 🧪 Reproducibility
- **Schema‑first**: Optionally define types in `configs/schema.json` and the loader will apply them; otherwise it infers.
- **Deterministic seeds** for splits and K‑Means init.
- **Makefile** commands for one‑shot runs; CI‑friendly.
- **No scikit‑learn for modeling** (Spark MLlib is used as required).

## 📊 Outputs
- `data/processed/` — cleaned parquet
- `data/features/` — `features.parquet` with engineered cols
- `models/` — persisted best model (used later in streaming/Part B)
- `reports/` — metrics JSON, ROC image(s)

## 🧯 Troubleshooting
- If Spark runs out of memory locally, reduce sample size or set `spark.executor.memory`/`spark.driver.memory` envs.
- macOS needs recent Java (Temurin 17 works well with PySpark 3.5).

## 📜 License
MIT — feel free to use and adapt.

---

**Author**: Kevin Tran  
**Portfolio**: (add URL)  •  **Email**: (add) 
