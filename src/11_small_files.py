import time
from pyspark.sql import SparkSession

spark = (
    SparkSession.builder
    .appName("OlistSmallFilesBenchmark")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

SOURCE = "hdfs://namenode:8020/datalake/bench/orders_scaled"
BASE = "hdfs://namenode:8020/datalake/bench"

MANY_FILES = f"{BASE}/orders_many_files"
FEW_FILES = f"{BASE}/orders_few_files"

# Load the scaled orders dataset
orders = spark.read.parquet(SOURCE)

print("Input order rows:", orders.count())

# Write with many output partitions
orders.repartition(200).write.mode("overwrite").parquet(MANY_FILES)

# Write with only 8 output partitions
orders.coalesce(8).write.mode("overwrite").parquet(FEW_FILES)

print("\nBoth Parquet layouts have been created.")

# Count rows and measure read time for the many-files layout
start = time.time()
many_count = spark.read.parquet(MANY_FILES).count()
many_time = time.time() - start

# Count rows and measure read time for the few-files layout
start = time.time()
few_count = spark.read.parquet(FEW_FILES).count()
few_time = time.time() - start

# Count actual Parquet data files
many_file_count = len([
    f for f in spark.sparkContext
    ._jvm.org.apache.hadoop.fs.FileSystem
    .get(spark._jsc.hadoopConfiguration())
    .listStatus(spark._jvm.org.apache.hadoop.fs.Path(MANY_FILES))
    if f.getPath().getName().endswith(".parquet")
])

few_file_count = len([
    f for f in spark.sparkContext
    ._jvm.org.apache.hadoop.fs.FileSystem
    .get(spark._jsc.hadoopConfiguration())
    .listStatus(spark._jvm.org.apache.hadoop.fs.Path(FEW_FILES))
    if f.getPath().getName().endswith(".parquet")
])

print("\n=== Small-Files Benchmark Results ===")
print(f"Many-files layout: {many_file_count} Parquet files")
print(f"Many-files row count: {many_count}")
print(f"Many-files read time: {many_time:.2f} seconds")
print(f"Few-files layout: {few_file_count} Parquet files")
print(f"Few-files row count: {few_count}")
print(f"Few-files read time: {few_time:.2f} seconds")

spark.stop()
