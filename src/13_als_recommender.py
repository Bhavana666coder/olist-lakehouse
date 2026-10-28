from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    count,
    countDistinct,
    lit,
    row_number,
    sum as spark_sum,
    when,
)
from pyspark.sql.window import Window
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

CLEANED_DIR = "hdfs://namenode:8020/datalake/curated/recommender_interactions"
OUTPUT_DIR = "hdfs://namenode:8020/datalake/curated/recommender_results"
K = 10

spark = (
    SparkSession.builder
    .appName("OlistALSRecommenderEvaluation")
    .getOrCreate()
)

print("\n========== OLIST RECOMMENDER EVALUATION ==========\n")

train_raw = spark.read.parquet(f"{CLEANED_DIR}/train")
test_raw = spark.read.parquet(f"{CLEANED_DIR}/test")

# Assign stable integer IDs using the union of training and test identifiers.
customers = (
    train_raw.select("customer_unique_id")
    .union(test_raw.select("customer_unique_id"))
    .distinct()
)

products = (
    train_raw.select("product_id")
    .union(test_raw.select("product_id"))
    .distinct()
)

customer_indexer = StringIndexer(
    inputCol="customer_unique_id",
    outputCol="user_id",
    handleInvalid="skip",
)

product_indexer = StringIndexer(
    inputCol="product_id",
    outputCol="item_id",
    handleInvalid="skip",
)

customer_model = customer_indexer.fit(customers)
product_model = product_indexer.fit(products)

customer_map = customer_model.transform(customers).select(
    "customer_unique_id", col("user_id").cast("int")
)

product_map = product_model.transform(products).select(
    "product_id", col("item_id").cast("int")
)

train = (
    train_raw
    .join(customer_map, on="customer_unique_id", how="inner")
    .join(product_map, on="product_id", how="inner")
    .select("user_id", "item_id")
    .distinct()
    .withColumn("rating", lit(1.0))
)

test = (
    test_raw
    .join(customer_map, on="customer_unique_id", how="inner")
    .join(product_map, on="product_id", how="inner")
    .select("user_id", "item_id")
    .distinct()
)

print("Training interactions:", train.count())
print("Test interactions:", test.count())
print("Training users:", train.select("user_id").distinct().count())
print("Training items:", train.select("item_id").distinct().count())

# Exclude training products from each user's recommendations.
seen_items = train.select("user_id", "item_id").withColumn(
    "seen", lit(1)
)

# Train implicit-feedback ALS.
als = ALS(
    userCol="user_id",
    itemCol="item_id",
    ratingCol="rating",
    implicitPrefs=True,
    rank=10,
    regParam=0.1,
    alpha=10.0,
    maxIter=10,
    nonnegative=True,
    coldStartStrategy="drop",
    seed=42,
)

model = als.fit(train)
print("ALS model training completed.")

# Generate candidates from the ALS model.
user_ids = test.select("user_id").distinct()
als_candidates = model.recommendForUserSubset(user_ids, K * 3)

als_recommendations = (
    als_candidates
    .select("user_id", "recommendations")
    .selectExpr(
        "user_id",
        "explode(recommendations) AS recommendation",
    )
    .select(
        "user_id",
        col("recommendation.item_id").alias("item_id"),
        col("recommendation.rating").alias("score"),
    )
    .join(seen_items, on=["user_id", "item_id"], how="left")
    .filter(col("seen").isNull())
)

als_rank_window = Window.partitionBy("user_id").orderBy(
    col("score").desc()
)

als_topk = (
    als_recommendations
    .withColumn("rank", row_number().over(als_rank_window))
    .filter(col("rank") <= K)
    .select("user_id", "item_id", "rank")
)

# Popularity baseline, computed only from training interactions.
popularity = (
    train.groupBy("item_id")
    .agg(count("*").alias("purchase_count"))
)

popularity_window = Window.orderBy(
    col("purchase_count").desc(), col("item_id").asc()
)

popular_topk = (
    popularity
    .withColumn("rank", row_number().over(popularity_window))
    .filter(col("rank") <= K)
    .select("item_id", "rank")
)

# Limit the popularity candidate list to avoid a huge user-product cross join.
# Each user receives up to 10 unseen items from the 100 most popular products.
popular_candidates_base = (
    popularity
    .orderBy(col("purchase_count").desc(), col("item_id").asc())
    .limit(100)
)

popular_candidates = (
    user_ids.crossJoin(popular_candidates_base)
    .join(seen_items, on=["user_id", "item_id"], how="left")
    .filter(col("seen").isNull())
)

popular_rank_window = Window.partitionBy("user_id").orderBy(
    col("purchase_count").desc(), col("item_id").asc()
)

popular_recommendations = (
    popular_candidates
    .withColumn("rank", row_number().over(popular_rank_window))
    .filter(col("rank") <= K)
    .select("user_id", "item_id", "rank")
)

# Evaluate ranking metrics over every eligible test user.
from pyspark.sql.functions import log

def evaluate(recommendations, method_name):
    # Include users even if they receive zero recommendations.
    evaluation_users = test.select("user_id").distinct()

    relevant = test.select("user_id", "item_id").distinct().withColumn(
        "is_relevant", lit(1)
    )

    matched = (
        recommendations
        .join(relevant, on=["user_id", "item_id"], how="left")
        .withColumn(
            "hit",
            when(col("is_relevant").isNotNull(), lit(1)).otherwise(lit(0)),
        )
        .withColumn(
            "discount",
            log(col("rank") + 1.0) / log(lit(2.0)),
        )
        .withColumn(
            "dcg",
            when(col("hit") == 1, 1.0 / col("discount"))
            .otherwise(lit(0.0)),
        )
    )

    per_user = matched.groupBy("user_id").agg(
        spark_sum("hit").alias("hits"),
        spark_sum("dcg").alias("dcg"),
        count("*").alias("recommendation_count"),
    )

    # Left join ensures users with no recommendations score zero.
    per_user = (
        evaluation_users
        .join(per_user, on="user_id", how="left")
        .fillna(
            0,
            subset=["hits", "dcg", "recommendation_count"],
        )
    )

    user_count = evaluation_users.count()

    if user_count == 0:
        raise ValueError("No eligible test users are available for evaluation.")

    metrics = per_user.agg(
        (
            spark_sum(when(col("hits") > 0, 1).otherwise(0))
            / lit(user_count)
        ).alias("hit_at_10"),
        (
            spark_sum("hits")
            / lit(user_count * K)
        ).alias("precision_at_10"),
        (
            spark_sum("dcg")
            / lit(user_count)
        ).alias("ndcg_at_10"),
    ).first()

    catalog_size = train.select("item_id").distinct().count()
    recommended_items = recommendations.select("item_id").distinct().count()
    coverage = (
        recommended_items / catalog_size
        if catalog_size else 0.0
    )

    return (
        method_name,
        float(metrics["hit_at_10"] or 0.0),
        float(metrics["precision_at_10"] or 0.0),
        float(metrics["ndcg_at_10"] or 0.0),
        float(coverage),
    )

als_metrics = evaluate(als_topk, "ALS")
popularity_metrics = evaluate(popular_recommendations, "Popularity")

results = spark.createDataFrame(
    [als_metrics, popularity_metrics],
    [
        "method",
        "hit_at_10",
        "precision_at_10",
        "ndcg_at_10",
        "catalog_coverage",
    ],
)

print("\n=== EVALUATION RESULTS ===")
results.show(truncate=False)

results.write.mode("overwrite").parquet(
    f"{OUTPUT_DIR}/metrics"
)

print("Results saved to:", f"{OUTPUT_DIR}/metrics")
print("\n========== EVALUATION COMPLETED ==========\n")

spark.stop()
