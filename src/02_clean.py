from pyspark.sql import SparkSession
from pathlib import Path

# Create Spark session
spark = (
    SparkSession.builder
    .appName("OlistDataCleaning")
    .getOrCreate()
)

# HDFS input and output directories
HDFS_RAW_DIR = "hdfs://namenode:8020/datalake/raw/olist"
HDFS_CLEANED_DIR = "hdfs://namenode:8020/datalake/cleaned"

# Names of the nine datasets
DATASETS = [
    "olist_customers_dataset",
    "olist_geolocation_dataset",
    "olist_order_items_dataset",
    "olist_order_payments_dataset",
    "olist_order_reviews_dataset",
    "olist_orders_dataset",
    "olist_products_dataset",
    "olist_sellers_dataset",
    "product_category_name_translation",
]

print("\n========== OLIST DATA CLEANING ==========\n")

for dataset in DATASETS:
    input_path = f"{HDFS_RAW_DIR}/{dataset}.csv"
    output_path = f"{HDFS_CLEANED_DIR}/{dataset}"

    print(f"\nProcessing: {dataset}")

    # Read CSV data
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(input_path)
    )

    rows_before = df.count()

    # Remove duplicate rows
    cleaned_df = df.dropDuplicates()

    rows_after = cleaned_df.count()

    # Save cleaned data in Parquet format
    (
        cleaned_df.write
        .mode("overwrite")
        .parquet(output_path)
    )

    print(f"Rows before cleaning: {rows_before}")
    print(f"Rows after cleaning:  {rows_after}")
    print(f"Saved to: {output_path}")

print("\n========== CLEANING COMPLETED ==========\n")

spark.stop()
