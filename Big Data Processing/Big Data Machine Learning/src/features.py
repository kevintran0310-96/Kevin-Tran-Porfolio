import os, yaml
from pyspark.sql import functions as F
from src.utils.io import spark

def main(cfg_path: str = "configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    s = spark("fraud-features")

    processed = cfg["data"]["processed_dir"]
    out_dir = cfg["data"]["features_dir"]
    os.makedirs(out_dir, exist_ok=True)

    br = s.read.parquet(os.path.join(processed, "browsing_behaviour.parquet"))
    cs = s.read.parquet(os.path.join(processed, "customer_session.parquet"))
    cust = s.read.parquet(os.path.join(processed, "customer.parquet"))
    fr = s.read.parquet(os.path.join(processed, "fraud_transaction.parquet"))

    levels = cfg["levels"]

    # Count actions per level
    def level_col(level_list):
        return F.sum(F.when(F.col("event_type").isin(level_list), 1).otherwise(0)).alias("tmp")

    agg = (
        br.groupBy("session_id")
          .agg(
              level_col(levels["L1"]).alias("L1_count"),
              level_col(levels["L2"]).alias("L2_count"),
              level_col(levels["L3"]).alias("L3_count"),
              F.count("*").alias("events_total"),
              F.min("event_time").alias("session_start"),
              F.max("event_time").alias("session_end"),
          )
    )

    # Ratios
    agg = agg.withColumn("L1_ratio", F.col("L1_count")/F.col("events_total"))
    agg = agg.withColumn("L2_ratio", F.col("L2_count")/F.col("events_total"))

    # Time-of-day bucket (approx using session mid-point)
    mid_ts = F.expr("(unix_timestamp(session_start)+unix_timestamp(session_end))/2")
    hour = F.hour(F.to_timestamp(mid_ts))
    agg = agg.withColumn("tod",
          F.when((hour >= 6) & (hour < 12), F.lit("morning"))
           .when((hour >= 12) & (hour < 18), F.lit("afternoon"))
           .when((hour >= 18) & (hour < 24), F.lit("evening"))
           .otherwise(F.lit("night"))
    )

    # Join customer via session
    feat = (
        agg.join(cs, "session_id", "left")
           .join(cust, "customer_id", "left")
    )

    # Attach label via transactions if present
    # Here we assume a join key 'transaction_id' present in customer_session; adjust as needed.
    if "transaction_id" in cs.columns and "transaction_id" in fr.columns:
        feat = feat.join(fr.select("transaction_id").withColumn("is_fraud", F.lit(1)), "transaction_id", "left")
    feat = feat.fillna({"is_fraud": 0})

    feat.write.mode("overwrite").parquet(os.path.join(out_dir, "features.parquet"))
    print("Features written →", os.path.join(out_dir, "features.parquet"))

if __name__ == "__main__":
    main()
