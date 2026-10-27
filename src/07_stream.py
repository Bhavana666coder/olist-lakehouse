import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, window, sum as spark_sum,
    count
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType
)

KAFKA_BOOTSTRAP_SERVERS = "172.18.0.1:9092"
KAFKA_TOPIC = "olist_orders"

RAW_OUTPUT = "hdfs://namenode:8020/datalake/stream/raw_events"
REVENUE_OUTPUT = "hdfs://namenode:8020/datalake/stream/revenue_per_minute"
RAW_CHECKPOINT = "hdfs://namenode:8020/datalake/stream/checkpoints/raw_events"
REVENUE_CHECKPOINT = "hdfs://namenode:8020/datalake/stream/checkpoints/revenue_per_minute"

schema = StructType([
    StructField("order_id", StringType(), True),
    StructField("order_item_id", IntegerType(), True),
    StructField("customer_id", StringType(), True),
    StructField("order_status", StringType(), True),
    StructField("order_purchase_timestamp", StringType(), True),
    StructField("product_id", StringType(), True),
    StructField("seller_id", StringType(), True),
    StructField("price", DoubleType(), True),
    StructField("freight_value", DoubleType(), True)
])

spark = (
    SparkSession.builder
    .appName("OlistKafkaStreaming")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("WARN")

try:
    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .load()
    )

    events = (
        kafka_df
        .select(from_json(col("value").cast("string"), schema).alias("event"))
        .select("event.*")
        .withColumn(
            "order_purchase_timestamp",
            to_timestamp(col("order_purchase_timestamp"))
        )
    )

    raw_query = (
        events.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", RAW_OUTPUT)
        .option("checkpointLocation", RAW_CHECKPOINT)
        .start()
    )

    valid_events = (
        events
        .filter(col("order_id").isNotNull())
        .filter(col("order_purchase_timestamp").isNotNull())
        .filter(col("price").isNotNull())
        .withWatermark("order_purchase_timestamp", "1 day")
        .dropDuplicates(["order_id", "order_item_id"])
    )

    revenue = (
        valid_events
        .groupBy(window(col("order_purchase_timestamp"), "1 minute"))
        .agg(
            spark_sum("price").alias("total_revenue"),
            count("*").alias("order_items")
        )
        .select(
            col("window.start").alias("minute_start"),
            col("window.end").alias("minute_end"),
            col("total_revenue"),
            col("order_items")
        )
    )

    revenue_query = (
        revenue.writeStream
        .format("parquet")
        .outputMode("append")
        .option("path", REVENUE_OUTPUT)
        .option("checkpointLocation", REVENUE_CHECKPOINT)
        .start()
    )

    print("Streaming started. Waiting for Kafka order events...")
    spark.streams.awaitAnyTermination()

finally:
    spark.stop()
