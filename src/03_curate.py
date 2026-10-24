from pyspark.sql import SparkSession
from pyspark.sql.functions import col, year, month, to_date

# 1. Create Spark session
spark = (
    SparkSession.builder
    .appName("OlistDataCuration")
    .getOrCreate()
)

# 2. Define HDFS directories
CLEANED_DIR = "hdfs://namenode:8020/datalake/cleaned"
CURATED_DIR = "hdfs://namenode:8020/datalake/curated"

print("\n========== OLIST DATA CURATION ==========\n")

# 3. Read cleaned datasets
orders = spark.read.parquet(f"{CLEANED_DIR}/olist_orders_dataset")
customers = spark.read.parquet(f"{CLEANED_DIR}/olist_customers_dataset")
payments = spark.read.parquet(f"{CLEANED_DIR}/olist_order_payments_dataset")

print("Cleaned datasets loaded successfully.")

# 4. Prepare order information
orders = orders.withColumn(
    "order_purchase_timestamp",
    to_date(col("order_purchase_timestamp"))
)

orders = (
    orders
    .withColumn("purchase_year", year(col("order_purchase_timestamp")))
    .withColumn("purchase_month", month(col("order_purchase_timestamp")))
)

# 5. Aggregate payments for each order
payment_summary = (
    payments
    .groupBy("order_id")
    .agg({"payment_value": "sum"})
    .withColumnRenamed("sum(payment_value)", "total_payment_value")
)

# 6. Join orders with customers
orders_customers = orders.join(
    customers,
    on="customer_id",
    how="left"
)

# 7. Join the payment summary
order_analytics = orders_customers.join(
    payment_summary,
    on="order_id",
    how="left"
)

# 8. Save the curated analytical dataset
output_path = f"{CURATED_DIR}/order_analytics"

(
    order_analytics.write
    .mode("overwrite")
    .parquet(output_path)
)

print(f"Curated order analytics saved to: {output_path}")
print(f"Total curated records: {order_analytics.count()}")

print("\n========== CURATION COMPLETED ==========\n")

spark.stop()
