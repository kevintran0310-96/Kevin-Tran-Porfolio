import os, yaml
from pyspark.sql import functions as F
from pyspark.sql.types import *
from src.utils.io import spark
from src.utils.schemas import load_schema

def main(cfg_path: str = "configs/config.yaml", use_sample: bool = True):
    cfg = yaml.safe_load(open(cfg_path))
    s = spark("fraud-etl")

    base = cfg["data"]["sample_dir"] if use_sample else cfg["data"]["raw_dir"]
    out = cfg["data"]["processed_dir"]
    os.makedirs(out, exist_ok=True)

    files = cfg["files"]

    def load_csv(name):
        schema = load_schema("configs/schema.json", name)
        reader = s.read.option("header", True)
        if schema is not None:
            reader = reader.schema(schema)
        else:
            reader = reader.option("inferSchema", True)
        return reader.csv(os.path.join(base, files.get(name, f"{name}.csv")))

    br = load_csv("browsing_behaviour")
    cs = load_csv("customer_session")
    cust = load_csv("customer")
    cat = load_csv("category")
    fr = load_csv("fraud_transaction")

    # Minimal cleaning: drop empty columns names, trim strings
    def trim_all(df):
        for c in df.columns:
            if dict(df.dtypes)[c] == "string":
                df = df.withColumn(c, F.trim(F.col(c)))
        return df

    br = trim_all(br)
    cs = trim_all(cs)
    cust = trim_all(cust)
    cat = trim_all(cat)
    fr = trim_all(fr)

    br.write.mode("overwrite").parquet(os.path.join(out, "browsing_behaviour.parquet"))
    cs.write.mode("overwrite").parquet(os.path.join(out, "customer_session.parquet"))
    cust.write.mode("overwrite").parquet(os.path.join(out, "customer.parquet"))
    cat.write.mode("overwrite").parquet(os.path.join(out, "category.parquet"))
    fr.write.mode("overwrite").parquet(os.path.join(out, "fraud_transaction.parquet"))

    print("ETL complete →", out)

if __name__ == "__main__":
    main()
