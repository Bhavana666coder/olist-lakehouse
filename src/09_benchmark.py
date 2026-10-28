import sys
import time

from pyspark.sql import SparkSession
from pyspark.sql.functions import broadcast, col, count, sum as spark_sum

# Read benchmark settings from command-line arguments
shuffle_partitions = int(sys.argv[1]) if len(sys.argv) > 1 else 200
use_broadcast = sys.argv[2].lower() == "true" if len(sys.argv) > 2 else False
use_aqe = sys.argv[3].lower() == "true" if len(sys.argv) > 3 else False

spark = (
    SparkSession.builder
    .appName("OlistSparkBenchmark")
    .config("spark.sql.shuffle.partitions", shuffle_partitions)
    .config("spark.sql.adaptive.enabled", str(use_aqe).lower())
    .config("spark.sql.autoBroadcastJoinThreshold", "-1")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

base = "hdfs://namenode:8020/datalake/bench"

# Load the scaled datasets
items = spark.read.parquet(f"{base}/order_items_scaled")
orders = spark.read.parquet(f"{base}/orders_scaled")
customers = spark.read.parquet(f"{base}/customers_scaled")

# Load dimension datasets from the original cleaned data
products = spark.read.parquet(
    "hdfs://namenode:8020/datalake/cleaned/olist_products_dataset"
)
sellers = spark.read.parquet(
    "hdfs://namenode:8020/datalake/cleaned/olist_sellers_dataset"
)

# Join order items to orders, customers, products, and sellers
joined = (
    items.join(orders, "order_id")
    .join(customers, "customer_id")
    .join(products, "product_id")
    .join(sellers, "seller_id")
)

# For the broadcast experiment, broadcast small dimension tables
if use_broadcast:
    joined = (
        items.join(orders, "order_id")
        .join(customers, "customer_id")
        .join(broadcast(products), "product_id")
        .join(broadcast(sellers), "seller_id")
    )

# Aggregate by customer state and product category
result = (
    joined.groupBy("customer_state", "product_category_name")
    .agg(
        count("*").alias("item_count"),
        spark_sum("price").alias("total_sales"),
    )
)

print(
    f"Configuration: shuffle_partitions={shuffle_partitions}, "
    f"broadcast={use_broadcast}, AQE={use_aqe}"
)

start = time.time()
row_count = result.count()
elapsed = time.time() - start

print(f"Aggregated result rows: {row_count}")
print(f"Benchmark execution time: {elapsed:.2f} seconds")

spark.stop()
