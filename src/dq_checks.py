from pyspark.sql import SparkSession
from pyspark.sql.functions import col

# Create Spark session
spark = (
    SparkSession.builder
    .appName("OlistDataQualityChecks")
    .getOrCreate()
)

# HDFS directory containing cleaned datasets
CLEANED_DIR = "hdfs://namenode:8020/datalake/cleaned"

# All nine cleaned datasets
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

try:
    print("\n========== DATA QUALITY CHECKS ==========\n", flush=True)

    for dataset in DATASETS:
        print(f"\n{'=' * 55}", flush=True)
        print(f"Dataset: {dataset}", flush=True)
        print(f"{'=' * 55}", flush=True)

        # Read the cleaned Parquet dataset from HDFS
        path = f"{CLEANED_DIR}/{dataset}"
        df = spark.read.parquet(path)

        # Count rows and columns
        total_rows = df.count()
        total_columns = len(df.columns)

        # Count duplicate rows
        duplicate_rows = total_rows - df.dropDuplicates().count()

        # Count missing values in every column
        missing_counts = {}

        for column in df.columns:
            missing_count = df.filter(col(column).isNull()).count()
            missing_counts[column] = missing_count

        # Calculate total missing values
        total_missing = sum(missing_counts.values())

        # Print dataset summary
        print(f"Total rows: {total_rows}", flush=True)
        print(f"Total columns: {total_columns}", flush=True)
        print(f"Duplicate rows: {duplicate_rows}", flush=True)
        print(f"Total missing values: {total_missing}", flush=True)

        # Print missing values by column
        print("\nMissing values by column:", flush=True)

        found_missing = False

        for column, count in missing_counts.items():
            if count > 0:
                print(f"  {column}: {count}", flush=True)
                found_missing = True

        if not found_missing:
            print("  No missing values found.", flush=True)

        # Report duplicate check
        if duplicate_rows == 0:
            print("Duplicate check: PASSED", flush=True)
        else:
            print("Duplicate check: WARNING", flush=True)

        # Report missing-value check
        if total_missing == 0:
            print("Missing value check: PASSED", flush=True)
        else:
            print("Missing value check: WARNING", flush=True)

    print(
        "\n========== QUALITY CHECKS COMPLETED ==========\n",
        flush=True
    )

finally:
    spark.stop()
