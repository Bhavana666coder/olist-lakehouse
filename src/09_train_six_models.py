from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer, StringIndexer, OneHotEncoder, VectorAssembler, MinMaxScaler
)
from pyspark.ml.classification import (
    LogisticRegression, LinearSVC, DecisionTreeClassifier,
    RandomForestClassifier, GBTClassifier, NaiveBayes
)
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator, RegressionEvaluator
)

spark = (
    SparkSession.builder
    .appName("OlistRepeatPurchaseSevenModelComparison")
    .getOrCreate()
)
spark.sparkContext.setLogLevel("ERROR")

FEATURE_PATH = "hdfs://namenode:8020/datalake/features/repeat_purchase"
MODEL_BASE = "hdfs://namenode:8020/datalake/models"
METRICS_PATH = "hdfs://namenode:8020/datalake/model_evaluation/repeat_purchase_v2"

numeric_features = [
    "item_count", "item_price_total", "freight_total", "seller_count",
    "avg_product_weight_g", "avg_product_photos_qty", "payment_value_total",
    "max_installments", "avg_review_score", "has_review_comment",
    "purchase_month", "purchase_dayofweek", "purchase_hour",
    "delivery_days", "delivery_delay_days", "is_late"
]
categorical_features = [
    "customer_state", "first_order_status", "payment_types", "product_category"
]

print("=" * 76)
print("OLIST REPEAT-PURCHASE PREDICTION: MODEL COMPARISON")
print("Six classifiers + one optional Linear Regression experiment")
print("=" * 76)

df = spark.read.parquet(FEATURE_PATH).withColumn("label", F.col("label").cast("double"))
required = numeric_features + categorical_features + ["label", "first_order_ts"]
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError("Missing required columns: " + ", ".join(missing))

for c in numeric_features:
    df = df.withColumn(c, F.col(c).cast("double"))

df = df.filter(
    F.col("label").isin(0.0, 1.0) & F.col("first_order_ts").isNotNull()
)
df = df.withColumn("_purchase_epoch", F.unix_timestamp("first_order_ts"))
df = df.filter(F.col("_purchase_epoch").isNotNull())

print("Usable records:", df.count())
print("Label distribution:")
df.groupBy("label").count().orderBy("label").show()

cutoff = df.select(
    F.expr("percentile_approx(_purchase_epoch, 0.8, 10000)").alias("cutoff")
).first()["cutoff"]
if cutoff is None:
    raise ValueError("Unable to calculate chronological cutoff.")

train = df.filter(F.col("_purchase_epoch") < cutoff)
test = df.filter(F.col("_purchase_epoch") >= cutoff)

train_count, test_count = train.count(), test.count()
if not train_count or not test_count:
    raise ValueError("Chronological split created an empty partition.")

positive = train.filter(F.col("label") == 1.0).count()
negative = train.filter(F.col("label") == 0.0).count()
if not positive or not negative:
    raise ValueError("Training set must contain both labels.")

total = positive + negative
positive_weight = total / (2.0 * positive)
negative_weight = total / (2.0 * negative)

train = train.withColumn(
    "classWeight",
    F.when(F.col("label") == 1.0, F.lit(positive_weight))
     .otherwise(F.lit(negative_weight))
)

print(f"Train records: {train_count}; test records: {test_count}")
print(f"Training labels: positive={positive}, negative={negative}")
print(f"Class weights: positive={positive_weight:.6f}, negative={negative_weight:.6f}")
print("Test label distribution:")
test.groupBy("label").count().orderBy("label").show()

def preprocessing_stages(for_naive_bayes=False):
    stages = []
    imputed = [c + "_imputed" for c in numeric_features]
    stages.append(Imputer(
        inputCols=numeric_features,
        outputCols=imputed,
        strategy="median"
    ))

    indexed = []
    for c in categorical_features:
        out = c + "_indexed"
        stages.append(StringIndexer(
            inputCol=c, outputCol=out, handleInvalid="keep"
        ))
        indexed.append(out)

    encoded = [c + "_encoded" for c in categorical_features]
    stages.append(OneHotEncoder(
        inputCols=indexed, outputCols=encoded, handleInvalid="keep"
    ))

    stages.append(VectorAssembler(
        inputCols=imputed + encoded,
        outputCol="features",
        handleInvalid="keep"
    ))

    if for_naive_bayes:
        # Multinomial Naive Bayes requires nonnegative feature values.
        # Scaling the assembled vector to [0, 1] handles any negative
        # numeric values (for example, negative delivery delay).
        stages.append(MinMaxScaler(
            inputCol="features", outputCol="nb_features"
        ))
    return stages

def evaluate_classifier(name, predictions):
    roc = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)
    pr = BinaryClassificationEvaluator(
        labelCol="label", rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    ).evaluate(predictions)

    row = predictions.agg(
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("tp"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("tn"),
        F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("fp"),
        F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("fn")
    ).first()

    tp, tn, fp, fn = [int(row[k] or 0) for k in ("tp", "tn", "fp", "fn")]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total_rows = tp + tn + fp + fn
    accuracy = (tp + tn) / total_rows if total_rows else 0.0

    result = {
        "model": name, "roc_auc": float(roc), "pr_auc": float(pr),
        "precision": float(precision), "recall": float(recall),
        "f1": float(f1), "accuracy": float(accuracy),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn
    }
    print("\n" + "-" * 76)
    print(name)
    for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"):
        print(f"{key}: {result[key]:.5f}")
    print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    return result

classification_results = []
model_specs = [
    ("Logistic Regression", "repeat_purchase_v2_logistic_regression",
     LogisticRegression(featuresCol="features", labelCol="label", weightCol="classWeight",
                        maxIter=50, regParam=0.01), False),
    ("Linear SVM", "repeat_purchase_v2_linear_svm",
     LinearSVC(featuresCol="features", labelCol="label", weightCol="classWeight",
               maxIter=50, regParam=0.01), False),
    ("Decision Tree", "repeat_purchase_v2_decision_tree",
     DecisionTreeClassifier(featuresCol="features", labelCol="label", weightCol="classWeight",
                            maxDepth=8, seed=42), False),
    ("Random Forest", "repeat_purchase_v2_random_forest",
     RandomForestClassifier(featuresCol="features", labelCol="label", weightCol="classWeight",
                            numTrees=50, maxDepth=8, seed=42), False),
    ("Gradient-Boosted Trees", "repeat_purchase_v2_gbt",
     GBTClassifier(featuresCol="features", labelCol="label", weightCol="classWeight",
                   maxIter=30, maxDepth=5, seed=42), False),
    ("Naive Bayes", "repeat_purchase_v2_naive_bayes",
     NaiveBayes(featuresCol="nb_features", labelCol="label", modelType="multinomial",
                smoothing=1.0), True),
]

for name, folder, estimator, use_nb in model_specs:
    print("\n" + "=" * 76)
    print("TRAINING:", name)
    print("=" * 76)
    pipeline = Pipeline(stages=preprocessing_stages(for_naive_bayes=use_nb) + [estimator])
    fitted = pipeline.fit(train)
    predictions = fitted.transform(test)
    result = evaluate_classifier(name, predictions)
    classification_results.append(result)

    model_path = MODEL_BASE + "/" + folder
    # These are new v2 paths, separate from the original model directories.
    fitted.write().overwrite().save(model_path)
    print("Saved model to:", model_path)

print("\n" + "=" * 76)
print("OPTIONAL EXPERIMENT: LINEAR REGRESSION ON A BINARY TARGET")
print("=" * 76)
print("This is included because it was requested, but it is not a standard")
print("classifier. Its predictions are evaluated with regression metrics.")
linreg = LinearRegression(
    featuresCol="features", labelCol="label", weightCol="classWeight",
    maxIter=50, regParam=0.01
)
linreg_pipeline = Pipeline(stages=preprocessing_stages() + [linreg])
linreg_model = linreg_pipeline.fit(train)
linreg_predictions = linreg_model.transform(test)

regression_metrics = {}
for metric in ("rmse", "mae", "r2"):
    evaluator = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName=metric
    )
    regression_metrics[metric] = float(evaluator.evaluate(linreg_predictions))
    print(f"Linear Regression {metric.upper()}: {regression_metrics[metric]:.5f}")

linreg_path = MODEL_BASE + "/repeat_purchase_v2_linear_regression_experiment"
linreg_model.write().overwrite().save(linreg_path)
print("Saved Linear Regression experiment to:", linreg_path)

# Save classifier comparison results to HDFS for reproducibility.
comparison = spark.createDataFrame(classification_results)
comparison.orderBy(F.desc("pr_auc")).show(truncate=False)
comparison.coalesce(1).write.mode("overwrite").option("header", True).csv(METRICS_PATH)
print("Classifier comparison saved to:", METRICS_PATH)

print("\n" + "=" * 76)
print("ALL MODEL RUNS COMPLETED")
print("Six classifiers saved under:", MODEL_BASE)
print("Linear Regression experiment saved under:", linreg_path)
print("Comparison metrics saved under:", METRICS_PATH)
print("=" * 76)

spark.stop()
