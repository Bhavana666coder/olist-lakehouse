import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, concat, lit

BASE = "hdfs://namenode:8020/datalake/cleaned"
OUTPUT = "hdfs://namenode:8020/datalake/bench"
REPLICATION_FACTOR = int(sys.argv[1]) if len(sys.argv) > 1 else 50

if REPLICATION_FACTOR < 1:
    raise ValueError("Replication factor must be at least 1")

spark = SparkSession.builder.appName("OlistScaleBenchmarkData").getOrCreate()
spark.sparkContext.setLogLevel("WARN")

replicas = spark.range(REPLICATION_FACTOR).withColumnRenamed("id", "replica")

items = spark.read.parquet(f"{BASE}/olist_order_items_dataset")
orders = spark.read.parquet(f"{BASE}/olist_orders_dataset")
customers = spark.read.parquet(f"{BASE}/olist_customers_dataset")

scaled_items = (
    items.crossJoin(replicas)
    .withColumn("order_id", concat(col("order_id"), lit("_rep"), col("replica")))
    .drop("replica")
)

scaled_orders = (
    orders.crossJoin(replicas)
    .withColumn("order_id", concat(col("order_id"), lit("_rep"), col("replica")))
    .withColumn("customer_id", concat(col("customer_id"), lit("_rep"), col("replica")))
    .drop("replica")
)

scaled_customers = (
    customers.crossJoin(replicas)
    .withColumn("customer_id", concat(col("customer_id"), lit("_rep"), col("replica")))
    .drop("replica")
)

scaled_items.write.mode("overwrite").parquet(f"{OUTPUT}/order_items_scaled")
scaled_orders.write.mode("overwrite").parquet(f"{OUTPUT}/orders_scaled")
scaled_customers.write.mode("overwrite").parquet(f"{OUTPUT}/customers_scaled")

print(f"Replication factor: {REPLICATION_FACTOR}")
print(f"Expected order-item rows: {112650 * REPLICATION_FACTOR}")
print(f"Expected order rows: {99441 * REPLICATION_FACTOR}")
print(f"Expected customer rows: {99441 * REPLICATION_FACTOR}")
print("Synthetic benchmark data written to:", OUTPUT)

spark.stop()
