from datetime import timedelta

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


# --------------------------------------------------
# 1. Start Spark and define HDFS locations
# --------------------------------------------------

spark = (
    SparkSession.builder
    .appName("OlistRepeatPurchaseFeatureEngineering")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

BASE = "hdfs://namenode:8020/datalake/cleaned"
OUTPUT = "hdfs://namenode:8020/datalake/features/repeat_purchase"

print("=" * 65)
print("OLIST REPEAT-PURCHASE FEATURE ENGINEERING")
print("=" * 65)


# --------------------------------------------------
# 2. Read the cleaned datasets
# --------------------------------------------------

orders = spark.read.parquet(f"{BASE}/olist_orders_dataset")
customers = spark.read.parquet(f"{BASE}/olist_customers_dataset")
items = spark.read.parquet(f"{BASE}/olist_order_items_dataset")
payments = spark.read.parquet(f"{BASE}/olist_order_payments_dataset")
reviews = spark.read.parquet(f"{BASE}/olist_order_reviews_dataset")
products = spark.read.parquet(f"{BASE}/olist_products_dataset")
translations = spark.read.parquet(
    f"{BASE}/product_category_name_translation"
)

print("All required datasets loaded.")


# --------------------------------------------------
# 3. Connect orders with unique customer identities
# --------------------------------------------------

customer_info = customers.select(
    "customer_id",
    "customer_unique_id",
    "customer_state",
    "customer_city"
)

all_orders = (
    orders
    .join(customer_info, on="customer_id", how="inner")
    .filter(F.col("order_purchase_timestamp").isNotNull())
    .select(
        "order_id",
        "customer_unique_id",
        "customer_state",
        "customer_city",
        "order_status",
        "order_purchase_timestamp",
        "order_delivered_customer_date",
        "order_estimated_delivery_date"
    )
)

max_purchase_timestamp = (
    all_orders
    .agg(F.max("order_purchase_timestamp").alias("max_ts"))
    .first()["max_ts"]
)

if max_purchase_timestamp is None:
    raise RuntimeError("No valid purchase timestamps were found.")

# A customer must have a full 180-day observation window.
cutoff_timestamp = max_purchase_timestamp - timedelta(days=180)

print("Latest purchase timestamp:", max_purchase_timestamp)
print("Eligibility cutoff:", cutoff_timestamp)


# --------------------------------------------------
# 4. Identify each customer's first order
# --------------------------------------------------

first_order_window = (
    Window
    .partitionBy("customer_unique_id")
    .orderBy(
        F.col("order_purchase_timestamp").asc(),
        F.col("order_id").asc()
    )
)

first_orders = (
    all_orders
    .withColumn(
        "row_number",
        F.row_number().over(first_order_window)
    )
    .filter(F.col("row_number") == 1)
    .select(
        "customer_unique_id",
        F.col("order_id").alias("first_order_id"),
        F.col("order_purchase_timestamp").alias("first_order_ts"),
        F.col("order_status").alias("first_order_status"),
        "customer_state",
        "customer_city",
        F.col("order_delivered_customer_date").alias(
            "first_delivered_ts"
        ),
        F.col("order_estimated_delivery_date").alias(
            "first_estimated_delivery_ts"
        )
    )
)

# Exclude customers whose first purchase is too recent to observe
# a complete 180-day repeat-purchase window.
eligible_first_orders = first_orders.filter(
    F.col("first_order_ts") <= F.lit(cutoff_timestamp)
)

print(
    "Customers with a complete observation window:",
    eligible_first_orders.count()
)


# --------------------------------------------------
# 5. Create the repeat-purchase label
# --------------------------------------------------

# Count subsequent orders within 180 days of the first purchase.
# Cancelled and unavailable orders do not count as repeat purchases.

first_keys = eligible_first_orders.select(
    "customer_unique_id",
    "first_order_ts"
).alias("f")

later_orders = all_orders.alias("l")

repeat_counts = (
    later_orders
    .join(
        first_keys,
        F.col("l.customer_unique_id") == F.col("f.customer_unique_id"),
        "inner"
    )
    .filter(
        (F.col("l.order_purchase_timestamp") > F.col("f.first_order_ts"))
        & (
            F.col("l.order_purchase_timestamp")
            <= F.col("f.first_order_ts") + F.expr("INTERVAL 180 DAYS")
        )
        & (~F.col("l.order_status").isin("canceled", "unavailable"))
    )
    .groupBy(F.col("f.customer_unique_id").alias("customer_unique_id"))
    .agg(
        F.countDistinct("l.order_id").alias("repeat_order_count")
    )
)

labeled_customers = (
    eligible_first_orders
    .join(repeat_counts, on="customer_unique_id", how="left")
    .fillna({"repeat_order_count": 0})
    .withColumn(
        "label",
        F.when(F.col("repeat_order_count") > 0, 1).otherwise(0)
    )
)


# --------------------------------------------------
# 6. Aggregate first-order item and product features
# --------------------------------------------------

item_details = (
    items
    .join(products, on="product_id", how="left")
    .join(translations, on="product_category_name", how="left")
    .withColumn(
        "category",
        F.coalesce(
            F.col("product_category_name_english"),
            F.col("product_category_name"),
            F.lit("unknown")
        )
    )
)

item_features = (
    item_details
    .groupBy("order_id")
    .agg(
        F.count("*").alias("item_count"),
        F.sum("price").alias("item_price_total"),
        F.sum("freight_value").alias("freight_total"),
        F.countDistinct("seller_id").alias("seller_count"),
        F.avg("product_weight_g").alias("avg_product_weight_g"),
        F.avg("product_photos_qty").alias("avg_product_photos_qty"),
        F.first("category", ignorenulls=True).alias("product_category")
    )
)


# --------------------------------------------------
# 7. Aggregate payment features
# --------------------------------------------------

payment_features = (
    payments
    .groupBy("order_id")
    .agg(
        F.sum("payment_value").alias("payment_value_total"),
        F.max("payment_installments").alias("max_installments"),
        F.concat_ws(
            "|",
            F.sort_array(F.collect_set("payment_type"))
        ).alias("payment_types")
    )
)


# --------------------------------------------------
# 8. Aggregate review features
# --------------------------------------------------

review_features = (
    reviews
    .groupBy("order_id")
    .agg(
        F.avg(
            F.col("review_score").cast("double")
        ).alias("avg_review_score"),
        F.max(
            F.when(
                F.length(
                    F.trim(F.coalesce(F.col("review_comment_message"), F.lit("")))
                ) > 0,
                1
            ).otherwise(0)
        ).alias("has_review_comment")
    )
)


# --------------------------------------------------
# 9. Combine the first-order features
# --------------------------------------------------

features = (
    labeled_customers
    .join(
        item_features,
        labeled_customers.first_order_id == item_features.order_id,
        "left"
    )
    .drop(item_features.order_id)
    .join(
        payment_features,
        labeled_customers.first_order_id == payment_features.order_id,
        "left"
    )
    .drop(payment_features.order_id)
    .join(
        review_features,
        labeled_customers.first_order_id == review_features.order_id,
        "left"
    )
    .drop(review_features.order_id)
)


# --------------------------------------------------
# 10. Calculate delivery and purchase-time features
# --------------------------------------------------

features = (
    features
    .withColumn(
        "purchase_month",
        F.month("first_order_ts")
    )
    .withColumn(
        "purchase_dayofweek",
        F.dayofweek("first_order_ts")
    )
    .withColumn(
        "purchase_hour",
        F.hour("first_order_ts")
    )
    .withColumn(
        "delivery_days",
        (
            F.unix_timestamp("first_delivered_ts")
            - F.unix_timestamp("first_order_ts")
        ) / 86400.0
    )
    .withColumn(
        "delivery_delay_days",
        (
            F.unix_timestamp("first_delivered_ts")
            - F.unix_timestamp("first_estimated_delivery_ts")
        ) / 86400.0
    )
    .withColumn(
        "is_late",
        F.when(
            F.col("first_delivered_ts").isNull()
            | F.col("first_estimated_delivery_ts").isNull(),
            F.lit(None).cast("double")
        ).when(
            F.col("first_delivered_ts")
            > F.col("first_estimated_delivery_ts"),
            1.0
        ).otherwise(0.0)
    )
)


# --------------------------------------------------
# 11. Select model-ready columns
# --------------------------------------------------

numeric_features = [
    "item_count",
    "item_price_total",
    "freight_total",
    "seller_count",
    "avg_product_weight_g",
    "avg_product_photos_qty",
    "payment_value_total",
    "max_installments",
    "avg_review_score",
    "has_review_comment",
    "purchase_month",
    "purchase_dayofweek",
    "purchase_hour",
    "delivery_days",
    "delivery_delay_days",
    "is_late"
]

categorical_features = [
    "customer_state",
    "first_order_status",
    "payment_types",
    "product_category"
]

selected_columns = [
    "customer_unique_id",
    "first_order_id",
    "first_order_ts",
    "label",
    "repeat_order_count"
] + numeric_features + categorical_features

model_data = features.select(*selected_columns)

# Remove duplicate customer rows if any unexpected join issue occurs.
model_data = model_data.dropDuplicates(["customer_unique_id"])


# --------------------------------------------------
# 12. Save the feature dataset in HDFS
# --------------------------------------------------

print("\nFinal feature dataset schema:")
model_data.printSchema()

total_customers = model_data.count()

print("Total eligible customers:", total_customers)

print("\nLabel distribution:")
model_data.groupBy("label").count().orderBy("label").show()

print("\nExample rows:")
model_data.select(
    "customer_unique_id",
    "first_order_id",
    "label",
    "repeat_order_count",
    "item_count",
    "item_price_total",
    "avg_review_score",
    "customer_state"
).show(10, truncate=False)

(
    model_data
    .write
    .mode("overwrite")
    .parquet(OUTPUT)
)

print("\nFeature dataset saved to:", OUTPUT)
print("Feature engineering completed successfully.")

spark.stop()
