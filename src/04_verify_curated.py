from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("VerifyCuratedOlistData")
    .getOrCreate()
)

path = "hdfs://namenode:8020/datalake/curated/order_analytics"

try:
    df = spark.read.parquet(path)

    print("\n========== CURATED DATA VERIFICATION ==========")

    print(f"\nTotal records: {df.count()}")

    print("\nColumns:")
    for column, dtype in df.dtypes:
        print(f"  {column}: {dtype}")

    print("\nSample records:")
    df.show(5, truncate=False)

    print("\n========== VERIFICATION COMPLETED ==========")

finally:
    spark.stop()
