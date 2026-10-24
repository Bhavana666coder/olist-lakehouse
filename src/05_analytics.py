from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    sum as spark_sum,
    avg,
    round as spark_round,
    desc
)

# Create Spark session
spark = (
    SparkSession.builder
    .appName("OlistAnalytics")
    .getOrCreate()
)

# HDFS input and output paths
CURATED_PATH = "hdfs://namenode:8020/datalake/curated/order_analytics"
ANALYTICS_PATH = "hdfs://namenode:8020/datalake/analytics"

try:
    print("\n========== OLIST ANALYTICS ==========\n", flush=True)

    # Read curated data
    df = spark.read.parquet(CURATED_PATH)

    print(f"Curated records loaded: {df.count()}", flush=True)

    # 1. Monthly sales
    monthly_sales = (
        df.filter(col("purchase_year").isNotNull())
        .filter(col("purchase_month").isNotNull())
        .groupBy("purchase_year", "purchase_month")
        .agg(
            spark_round(
                spark_sum("total_payment_value"), 2
            ).alias("total_sales"),
            count("order_id").alias("order_count")
        )
        .orderBy("purchase_year", "purchase_month")
    )

    monthly_sales.write.mode("overwrite").parquet(
        f"{ANALYTICS_PATH}/monthly_sales"
    )

    print("\n--- Monthly Sales (first 12 rows) ---", flush=True)
    monthly_sales.show(12, truncate=False)

    # 2. Order status analysis
    status_analysis = (
        df.groupBy("order_status")
        .agg(count("order_id").alias("order_count"))
        .orderBy(desc("order_count"))
    )

    status_analysis.write.mode("overwrite").parquet(
        f"{ANALYTICS_PATH}/order_status"
    )

    print("\n--- Orders by Status ---", flush=True)
    status_analysis.show(truncate=False)

    # 3. Top 10 customer cities by order count
    city_analysis = (
        df.groupBy("customer_city")
        .agg(count("order_id").alias("order_count"))
        .orderBy(desc("order_count"))
    )

    city_analysis.write.mode("overwrite").parquet(
        f"{ANALYTICS_PATH}/top_customer_cities"
    )

    print("\n--- Top 10 Customer Cities ---", flush=True)
    city_analysis.show(10, truncate=False)

    # 4. Overall payment analysis
    payment_analysis = df.agg(
        spark_round(
            spark_sum("total_payment_value"), 2
        ).alias("total_payment_value"),
        spark_round(
            avg("total_payment_value"), 2
        ).alias("average_payment_per_order"),
        count("order_id").alias("total_orders")
    )

    payment_analysis.write.mode("overwrite").parquet(
        f"{ANALYTICS_PATH}/payment_summary"
    )

    print("\n--- Overall Payment Summary ---", flush=True)
    payment_analysis.show(truncate=False)

    # 5. Yearly sales
    yearly_sales = (
        df.filter(col("purchase_year").isNotNull())
        .groupBy("purchase_year")
        .agg(
            spark_round(
                spark_sum("total_payment_value"), 2
            ).alias("total_sales"),
            count("order_id").alias("order_count")
        )
        .orderBy("purchase_year")
    )

    yearly_sales.write.mode("overwrite").parquet(
        f"{ANALYTICS_PATH}/yearly_sales"
    )

    print("\n--- Yearly Sales ---", flush=True)
    yearly_sales.show(truncate=False)

    print(
        f"\nAnalytics results saved under: {ANALYTICS_PATH}",
        flush=True
    )

    print("\n========== ANALYTICS COMPLETED ==========\n", flush=True)

finally:
    spark.stop()
