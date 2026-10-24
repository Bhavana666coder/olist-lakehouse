from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("olist_hdfs_test").getOrCreate()

path = "hdfs://namenode:8020/datalake/raw/olist/olist_customers_dataset.csv"

df = (
    spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(path)
)

print("CUSTOMER ROW COUNT:", df.count())
print("CUSTOMER COLUMNS:", df.columns)

df.show(5, truncate=False)

spark.stop()
