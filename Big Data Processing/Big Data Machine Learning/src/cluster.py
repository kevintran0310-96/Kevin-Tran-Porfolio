import os, yaml
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from src.utils.io import spark

def main(cfg_path: str = "configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    s = spark("fraud-cluster")

    feats = s.read.parquet(os.path.join(cfg["data"]["features_dir"], "features.parquet"))
    numeric = ["L1_count","L2_count","L3_count","L1_ratio","L2_ratio"]
    input_cols = [c for c in numeric if c in feats.columns]

    vec = VectorAssembler(inputCols=input_cols, outputCol="features")
    data = vec.transform(feats).select("features")

    kmin, kmax = int(cfg["kmeans"]["k_min"]), int(cfg["kmeans"]["k_max"])
    for k in range(kmin, kmax+1):
        km = KMeans(k=k, seed=cfg["model"]["seed"], featuresCol="features")
        model = km.fit(data)
        wssse = model.summary.trainingCost
        print(f"K={k} WSSSE={wssse:.2f}")

if __name__ == "__main__":
    main()
