from pyspark.sql import SparkSession
from pyspark import SparkConf

def spark(app_name: str = "fraud-etl"):
    conf = (
        SparkConf()
        .setAppName(app_name)
        .set("spark.sql.shuffle.partitions", "200")
        .set("spark.sql.files.maxPartitionBytes", str(16 * 1024 * 1024))  # <=16MB
        .set("spark.driver.memory", "4g")
        .set("spark.executor.memory", "4g")
        .setMaster("local[*]")
    )
    return SparkSession.builder.config(conf=conf).getOrCreate()
