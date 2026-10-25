from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import (
    Imputer,
    StringIndexer,
    OneHotEncoder,
    VectorAssembler
)
from pyspark.ml.classification import (
    LogisticRegression,
    RandomForestClassifier
)
from pyspark.ml.evaluation import BinaryClassificationEvaluator


# ============================================================
# 1. START SPARK
# ============================================================

spark = (
    SparkSession.builder
    .appName("OlistRepeatPurchaseModelTraining")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

FEATURE_PATH = (
    "hdfs://namenode:8020/datalake/features/repeat_purchase"
)

MODEL_BASE = "hdfs://namenode:8020/datalake/models"

print("=" * 70)
print("OLIST REPEAT-PURCHASE PREDICTION")
print("MODEL TRAINING AND EVALUATION")
print("=" * 70)


# ============================================================
# 2. LOAD FEATURE DATASET
# ============================================================

df = spark.read.parquet(FEATURE_PATH)

# Make sure the target is numeric.
df = df.withColumn("label", F.col("label").cast("double"))

print("\nTotal feature records:", df.count())

print("\nDataset schema:")
df.printSchema()

print("\nOverall label distribution:")
df.groupBy("label").count().orderBy("label").show()


# ============================================================
# 3. DEFINE INPUT FEATURES
# ============================================================

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

# IDs, timestamps, and the target-related count are not
# supplied to the model as predictive features.
required_columns = (
    numeric_features
    + categorical_features
    + ["label", "first_order_ts"]
)

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Required columns missing from feature dataset: "
        + ", ".join(missing_columns)
    )

# Cast numeric inputs to double for consistent preprocessing.
for column in numeric_features:
    df = df.withColumn(
        column,
        F.col(column).cast("double")
    )

# Remove records without a usable label or first-order timestamp.
df = df.filter(
    F.col("label").isin(0.0, 1.0)
    & F.col("first_order_ts").isNotNull()
)

# Exclude null or invalid timestamps from the temporal split.
df = df.withColumn(
    "_purchase_epoch",
    F.unix_timestamp("first_order_ts")
)

df = df.filter(F.col("_purchase_epoch").isNotNull())

print("\nRecords available for training:", df.count())


# ============================================================
# 4. CHRONOLOGICAL TRAIN/TEST SPLIT
# ============================================================

# Use the 80th percentile of first-order timestamps as the
# cutoff. Earlier records are training data; later records
# are test data.

cutoff_row = df.select(
    F.expr(
        "percentile_approx(_purchase_epoch, 0.8, 10000)"
    ).alias("cutoff")
).first()

cutoff = cutoff_row["cutoff"]

if cutoff is None:
    raise ValueError("Unable to calculate chronological cutoff.")

train_df = df.filter(F.col("_purchase_epoch") < cutoff)
test_df = df.filter(F.col("_purchase_epoch") >= cutoff)

train_count = train_df.count()
test_count = test_df.count()

print("\nChronological split:")
print("Training records:", train_count)
print("Testing records:", test_count)

if train_count == 0 or test_count == 0:
    raise ValueError(
        "The chronological split produced an empty dataset."
    )

print("\nTraining label distribution:")
train_df.groupBy("label").count().orderBy("label").show()

print("\nTesting label distribution:")
test_df.groupBy("label").count().orderBy("label").show()

train_positive = train_df.filter(F.col("label") == 1.0).count()
train_negative = train_df.filter(F.col("label") == 0.0).count()

if train_positive == 0 or train_negative == 0:
    raise ValueError(
        "Training data must contain both positive and negative labels."
    )


# ============================================================
# 5. HANDLE CLASS IMBALANCE
# ============================================================

# Repeat purchases are much less common than non-repeat purchases.
# Calculate balanced class weights from the training set only.

total_train = train_positive + train_negative

positive_weight = total_train / (2.0 * train_positive)
negative_weight = total_train / (2.0 * train_negative)

train_df = train_df.withColumn(
    "classWeight",
    F.when(
        F.col("label") == 1.0,
        F.lit(positive_weight)
    ).otherwise(F.lit(negative_weight))
)

print("\nClass weights:")
print("Positive class weight:", positive_weight)
print("Negative class weight:", negative_weight)


# ============================================================
# 6. BUILD THE PREPROCESSING PIPELINE
# ============================================================

def build_preprocessing_stages():
    stages = []

    # Replace missing numeric values with the median.
    imputed_features = [
        column + "_imputed"
        for column in numeric_features
    ]

    imputer = Imputer(
        inputCols=numeric_features,
        outputCols=imputed_features,
        strategy="median"
    )

    stages.append(imputer)

    # Convert categorical values into numeric category indices.
    indexed_features = []

    for column in categorical_features:
        indexed_column = column + "_indexed"

        indexer = StringIndexer(
            inputCol=column,
            outputCol=indexed_column,
            handleInvalid="keep"
        )

        stages.append(indexer)
        indexed_features.append(indexed_column)

    # Convert category indices into one-hot vectors.
    encoded_features = [
        column + "_encoded"
        for column in categorical_features
    ]

    encoder = OneHotEncoder(
        inputCols=indexed_features,
        outputCols=encoded_features,
        handleInvalid="keep"
    )

    stages.append(encoder)

    # Combine all processed inputs into one feature vector.
    assembler = VectorAssembler(
        inputCols=imputed_features + encoded_features,
        outputCol="features",
        handleInvalid="keep"
    )

    stages.append(assembler)

    return stages


# ============================================================
# 7. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, model, predictions):
    print("\n" + "=" * 70)
    print(model_name, "- EVALUATION")
    print("=" * 70)

    roc_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    )

    pr_evaluator = BinaryClassificationEvaluator(
        labelCol="label",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderPR"
    )

    roc_auc = roc_evaluator.evaluate(predictions)
    pr_auc = pr_evaluator.evaluate(predictions)

    # Calculate classification metrics at the model's default
    # prediction threshold. These metrics may be affected by
    # class imbalance and should be interpreted alongside PR-AUC.

    counts = predictions.agg(
        F.sum(
            F.when(
                (F.col("label") == 1.0)
                & (F.col("prediction") == 1.0),
                1
            ).otherwise(0)
        ).alias("tp"),

        F.sum(
            F.when(
                (F.col("label") == 0.0)
                & (F.col("prediction") == 0.0),
                1
            ).otherwise(0)
        ).alias("tn"),

        F.sum(
            F.when(
                (F.col("label") == 0.0)
                & (F.col("prediction") == 1.0),
                1
            ).otherwise(0)
        ).alias("fp"),

        F.sum(
            F.when(
                (F.col("label") == 1.0)
                & (F.col("prediction") == 0.0),
                1
            ).otherwise(0)
        ).alias("fn")
    ).first()

    tp = int(counts["tp"] or 0)
    tn = int(counts["tn"] or 0)
    fp = int(counts["fp"] or 0)
    fn = int(counts["fn"] or 0)

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0

    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")

    print("\nConfusion matrix counts:")
    print("True positives :", tp)
    print("True negatives :", tn)
    print("False positives:", fp)
    print("False negatives:", fn)

    print("\nPredicted label distribution:")
    predictions.groupBy("prediction").count().orderBy(
        "prediction"
    ).show()

    return {
        "model": model_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }


# ============================================================
# 8. TRAIN LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 70)
print("TRAINING LOGISTIC REGRESSION")
print("=" * 70)

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    maxIter=50,
    regParam=0.01
)

lr_pipeline = Pipeline(
    stages=build_preprocessing_stages() + [lr]
)

lr_model = lr_pipeline.fit(train_df)

lr_predictions = lr_model.transform(test_df)

lr_results = evaluate_model(
    "Logistic Regression",
    lr_model,
    lr_predictions
)

lr_model_path = (
    MODEL_BASE + "/repeat_purchase_logistic_regression"
)

lr_model.write().overwrite().save(lr_model_path)

print("\nLogistic Regression model saved to:")
print(lr_model_path)


# ============================================================
# 9. TRAIN RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

rf = RandomForestClassifier(
    featuresCol="features",
    labelCol="label",
    weightCol="classWeight",
    numTrees=50,
    maxDepth=8,
    seed=42
)

rf_pipeline = Pipeline(
    stages=build_preprocessing_stages() + [rf]
)

rf_model = rf_pipeline.fit(train_df)

rf_predictions = rf_model.transform(test_df)

rf_results = evaluate_model(
    "Random Forest",
    rf_model,
    rf_predictions
)

rf_model_path = (
    MODEL_BASE + "/repeat_purchase_random_forest"
)

rf_model.write().overwrite().save(rf_model_path)

print("\nRandom Forest model saved to:")
print(rf_model_path)


# ============================================================
# 10. COMPARE THE MODELS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

comparison = spark.createDataFrame(
    [lr_results, rf_results]
)

comparison.select(
    "model",
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "accuracy"
).show(truncate=False)

print("\nMODEL TRAINING COMPLETED SUCCESSFULLY.")
print("Saved models under:", MODEL_BASE)

spark.stop()
