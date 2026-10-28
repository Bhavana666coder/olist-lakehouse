from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    countDistinct,
    max as spark_max,
    row_number,
)
from pyspark.sql.window import Window

CLEANED_DIR = "hdfs://namenode:8020/datalake/cleaned"
OUTPUT_DIR = "hdfs://namenode:8020/datalake/curated/recommender_interactions"

spark = (
    SparkSession.builder
    .appName("OlistRecommenderInteractions")
    .getOrCreate()
)

print("\n========== RECOMMENDER INTERACTION BUILD ==========\n")

items = spark.read.parquet(
    f"{CLEANED_DIR}/olist_order_items_dataset"
)
orders = spark.read.parquet(
    f"{CLEANED_DIR}/olist_orders_dataset"
)
customers = spark.read.parquet(
    f"{CLEANED_DIR}/olist_customers_dataset"
)

print("=== Order items schema ===")
items.printSchema()
print("=== Orders schema ===")
orders.printSchema()
print("=== Customers schema ===")
customers.printSchema()

# Retain delivered orders with a valid purchase timestamp.
delivered_orders = (
    orders
    .filter(
        (col("order_status") == "delivered")
        & col("order_purchase_timestamp").isNotNull()
    )
    .select("order_id", "customer_id", "order_purchase_timestamp")
)

# Join order items to orders and stable customer identifiers.
raw_interactions = (
    items
    .join(delivered_orders, on="order_id", how="inner")
    .join(
        customers.select("customer_id", "customer_unique_id"),
        on="customer_id",
        how="inner",
    )
    .filter(
        col("customer_unique_id").isNotNull()
        & col("product_id").isNotNull()
    )
    .select(
        "customer_unique_id",
        "product_id",
        "order_id",
        "order_purchase_timestamp",
        "order_item_id",
    )
)

print("Delivered order-item records:", raw_interactions.count())

# Aggregate repeated purchases of the same product by the same customer.
# Keep the latest purchase timestamp for the chronological train/test split.
interactions = (
    raw_interactions
    .groupBy("customer_unique_id", "product_id")
    .agg(
        spark_max("order_purchase_timestamp").alias(
            "last_purchase_timestamp"
        ),
    )
)

# Count distinct products per customer and customers per product.
customer_counts = interactions.groupBy("customer_unique_id").agg(
    countDistinct("product_id").alias("distinct_products")
)
product_counts = interactions.groupBy("product_id").agg(
    countDistinct("customer_unique_id").alias("distinct_customers")
)

eligible_customers = customer_counts.filter(
    col("distinct_products") >= 2
).select("customer_unique_id")

eligible_products = product_counts.filter(
    col("distinct_customers") >= 2
).select("product_id")

filtered = (
    interactions
    .join(eligible_customers, on="customer_unique_id", how="inner")
    .join(eligible_products, on="product_id", how="inner")
)

print("Unique customer-product interactions after filtering:", filtered.count())

# Hold out each customer's most recently purchased product.
customer_window = Window.partitionBy("customer_unique_id").orderBy(
    col("last_purchase_timestamp").desc(),
    col("product_id").desc(),
)

ranked = filtered.withColumn(
    "interaction_rank",
    row_number().over(customer_window),
)

train = (
    ranked
    .filter(col("interaction_rank") > 1)
    .drop("interaction_rank")
)

test = (
    ranked
    .filter(col("interaction_rank") == 1)
    .drop("interaction_rank")
    .select("customer_unique_id", "product_id")
)

# Ensure every test customer has training history.
train_customers = train.select("customer_unique_id").distinct()
test = test.join(train_customers, on="customer_unique_id", how="inner")

print("Training interactions:", train.count())
print("Test interactions:", test.count())
print(
    "Training customers:",
    train.select("customer_unique_id").distinct().count(),
)
print(
    "Test customers:",
    test.select("customer_unique_id").distinct().count(),
)
print("Training products:", train.select("product_id").distinct().count())

train_path = f"{OUTPUT_DIR}/train"
test_path = f"{OUTPUT_DIR}/test"

train.write.mode("overwrite").parquet(train_path)
test.write.mode("overwrite").parquet(test_path)

print("\nTraining data saved to:", train_path)
print("Test data saved to:", test_path)
print("\n========== INTERACTION BUILD COMPLETED ==========\n")

spark.stop()
