from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import time

spark = (
    SparkSession.builder
    .appName("OlistPartitionPruningBenchmark")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

BASE = "hdfs://namenode:8020/datalake/bench"
SOURCE = "hdfs://namenode:8020/datalake/cleaned/olist_orders_dataset"
PARTITIONED = f"{BASE}/orders_partitioned"
UNPARTITIONED = f"{BASE}/orders_unpartitioned"

# Load the original orders dataset
orders = spark.read.parquet(SOURCE)

# Create an unpartitioned Parquet dataset
orders.write.mode("overwrite").parquet(UNPARTITIONED)

# Create a partitioned Parquet dataset using order_status
orders.write.mode("overwrite").partitionBy("order_status").parquet(PARTITIONED)

print("Both datasets have been created.")

# Test the same filter on the unpartitioned dataset
start = time.time()
unpartitioned_count = (
    spark.read.parquet(UNPARTITIONED)
    .filter(col("order_status") == "delivered")
    .count()
)
unpartitioned_time = time.time() - start

# Test the same filter on the partitioned dataset
start = time.time()
partitioned_count = (
    spark.read.parquet(PARTITIONED)
    .filter(col("order_status") == "delivered")
    .count()
)
partitioned_time = time.time() - start

print("\n=== Partition Pruning Results ===")
print(f"Unpartitioned rows: {unpartitioned_count}")
print(f"Unpartitioned read time: {unpartitioned_time:.2f} seconds")
print(f"Partitioned rows: {partitioned_count}")
print(f"Partitioned read time: {partitioned_time:.2f} seconds")

spark.stop()
