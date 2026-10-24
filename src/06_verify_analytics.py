from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("VerifyAnalytics")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

base = "hdfs://namenode:8020/datalake/analytics"

datasets = [
    "monthly_sales",
    "order_status",
    "top_customer_cities",
    "payment_summary",
    "yearly_sales"
]

try:
    for name in datasets:
        print("\n========== " + name + " ==========", flush=True)

        df = spark.read.parquet(base + "/" + name)

        print("Columns:", df.columns, flush=True)
        print("Number of rows:", df.count(), flush=True)
        df.show(12, truncate=False)

    print("\nALL ANALYTICS OUTPUTS VERIFIED", flush=True)

finally:
    spark.stop()
