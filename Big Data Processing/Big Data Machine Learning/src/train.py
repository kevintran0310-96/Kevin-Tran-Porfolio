import os, yaml
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, OneHotEncoder
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql import functions as F
from src.utils.io import spark

def main(cfg_path: str = "configs/config.yaml"):
    cfg = yaml.safe_load(open(cfg_path))
    s = spark("fraud-train")

    feats = s.read.parquet(os.path.join(cfg["data"]["features_dir"], "features.parquet"))

    label_col = cfg["model"]["label_col"]
    seed = int(cfg["model"]["seed"])

    # Choose a small set of numeric features for sample run
    numeric = ["L1_count","L2_count","L3_count","L1_ratio","L2_ratio"]
    categorical = ["tod"] if "tod" in feats.columns else []

    stages = []
    if categorical:
        indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical]
        encoders = [OneHotEncoder(inputCol=f"{c}_idx", outputCol=f"{c}_oh") for c in categorical]
        stages.extend(indexers + encoders)
        input_vec = numeric + [f"{c}_oh" for c in categorical]
    else:
        input_vec = numeric

    assembler = VectorAssembler(inputCols=input_vec, outputCol="features")
    rf = RandomForestClassifier(labelCol=label_col, featuresCol="features", seed=seed)
    gbt = GBTClassifier(labelCol=label_col, featuresCol="features", seed=seed, maxIter=30)

    # Split
    train, test = feats.randomSplit([1.0 - cfg["model"]["test_size"], cfg["model"]["test_size"]], seed=seed)

    def fit_eval(model, name):
        pipe = Pipeline(stages=stages + [assembler, model])
        m = pipe.fit(train)
        pred = m.transform(test)
        evaluator = BinaryClassificationEvaluator(labelCol=label_col, rawPredictionCol="rawPrediction", metricName="areaUnderROC")
        auc = evaluator.evaluate(pred)
        # simple confusion-ish counts
        cm = pred.select(
            F.sum(F.expr(f"case when {label_col}=1 and prediction=1 then 1 else 0 end")).alias("TP"),
            F.sum(F.expr(f"case when {label_col}=0 and prediction=0 then 1 else 0 end")).alias("TN"),
            F.sum(F.expr(f"case when {label_col}=0 and prediction=1 then 1 else 0 end")).alias("FP"),
            F.sum(F.expr(f"case when {label_col}=1 and prediction=0 then 1 else 0 end")).alias("FN"),
        ).first().asDict()
        print(f"{name}: AUC={auc:.3f}, CM={cm}")
        return name, auc, m

    results = []
    for model, name in [(rf, "RF"), (gbt, "GBT")]:
        results.append(fit_eval(model, name))

    # Persist best
    best = max(results, key=lambda x: x[1])
    out_dir = os.path.join(cfg["data"]["models_dir"], best[0])
    best[2].write().overwrite().save(out_dir)
    print("Saved best model →", out_dir)

if __name__ == "__main__":
    main()
