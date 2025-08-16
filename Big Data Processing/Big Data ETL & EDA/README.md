# Big Data ETL & EDA Project

## 📌 Overview
This project demonstrates an **end-to-end Big Data pipeline** for **ETL (Extract, Transform, Load)** and **EDA (Exploratory Data Analysis)**.  

The project is based on a large-scale **e-commerce transactions dataset**. The motivation was to simulate how real-world companies—like online retailers and logistics providers—can process raw transactional data at scale, clean and transform it into a reliable format, and then perform exploratory analysis to uncover business insights.  

I created this project as part of my **Data Science portfolio**, with the goal of showcasing my skills in:  
- **Data engineering** (scalable ETL workflows with PySpark)  
- **Data analysis** (statistical exploration, visualization, and insights)  
- **Business thinking** (how data can drive revenue optimization and operational improvements).  

---

## ⚙️ Tech Stack
- **Language**: Python (pandas, PySpark)  
- **Data Processing**: Apache Spark / PySpark  
- **Storage**: CSV (sample transactions dataset)  
- **Visualization**: Matplotlib, Seaborn  
- **Environment**: Jupyter Notebook & local development  
- **Version Control**: Git & GitHub  

---

## 📂 Project Structure
```
Big-Data-ETL-EDA-Project/
│
├── data/                  # Sample transaction data (lightweight subset)
├── notebooks/             # Jupyter notebooks for ETL and EDA
│   ├── 1_data_cleaning.ipynb
│   ├── 2_etl_pipeline.ipynb
│   ├── 3_eda_analysis.ipynb
│
├── src/                   # Python source code (modular ETL scripts)
│   ├── __init__.py
│   ├── etl.py
│   └── utils.py
│
├── requirements.txt       # Project dependencies
├── README.md              # Project documentation
└── LICENSE
```

---

## 🚀 ETL Pipeline
1. **Extract**  
   - Load raw e-commerce transactions dataset (CSV).  
   - Define schema and ingest using Spark.  

2. **Transform**  
   - Clean missing values and invalid records.  
   - Normalize categorical and time-based fields.  
   - Create new features like **order revenue**, **profit margins**, and **customer segments**.  

3. **Load**  
   - Export clean dataset to structured CSV/Parquet.  
   - Provide reusable outputs for analytics or dashboards.  

---

## 📊 Exploratory Data Analysis
The EDA is focused on answering questions relevant to **business decision-making**:  
- Which **product categories** drive the most revenue and profit?  
- What are the **seasonal or monthly sales trends**?  
- Who are the **high-value customers** based on purchasing behavior?  
- Are there any **data quality issues** (duplicates, invalid entries) impacting decisions?  

---

## 📝 Results & Findings
- Found **peak sales periods** (e.g., holiday months).  
- Identified categories with **high revenue but low profit margins**.  
- Detected **duplicates and invalid transactions**, suggesting a need for stronger data governance.  
- Highlighted **customer buying patterns**, useful for targeted marketing.  

---

## 💡 Learning Outcomes
Through this project, I gained practical experience in:  
- Handling **large, messy datasets** and making them analysis-ready.  
- Designing an ETL pipeline that is **modular and scalable**.  
- Using EDA to translate raw data into **actionable business insights**.  
- Structuring projects for **clarity and reusability** (important for team collaboration).  

---

## 🔎 Use Case Scenario
Imagine an **online retail company** experiencing rapid growth. They collect millions of transaction records daily but struggle with:  
- **Dirty data** (missing or duplicated records).  
- **Limited visibility** into profit margins by product category.  
- **Inefficient marketing** because customer segmentation is unclear.  

By applying the **ETL & EDA workflow** from this project, the company can:  
- **Automate data cleaning and transformation** so analysts always work with reliable data.  
- **Identify top-selling products** and categories that maximize profit.  
- **Detect seasonal sales patterns**, improving demand forecasting and inventory planning.  
- **Segment customers** based on behavior, enabling personalized promotions and higher retention.  

This directly translates into **better decision-making**, **cost savings**, and **higher revenue growth**.  

---

## 🔧 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/Big-Data-ETL-EDA-Project.git
   cd Big-Data-ETL-EDA-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run ETL pipeline:
   ```bash
   python src/etl.py
   ```
4. Open Jupyter Notebook for EDA:
   ```bash
   jupyter notebook notebooks/3_eda_analysis.ipynb
   ```

---

## 📌 Future Improvements
- Connect to a **cloud data warehouse** (AWS Redshift, BigQuery).  
- Orchestrate pipeline with **Airflow or Prefect**.  
- Build **real-time dashboards** in Tableau/Power BI.  
- Extend dataset to simulate **millions of rows** for true big data scale.  

---

## 👤 Author
**Kevin Tran**  
- 💼 Data Scientist | E-commerce & Logistics Experience  
- 🌐 [Portfolio Website](#)  
- 📧 [Your Email]  
