from pyspark.sql import SparkSession, functions as F
from pyspark.ml import Pipeline
from pyspark.ml.feature import Imputer, StringIndexer, OneHotEncoder, VectorAssembler, SQLTransformer
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

spark = (
    SparkSession.builder
    .appName("OlistFinishNaiveBayesAndLinearRegression")
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

df = spark.read.parquet(FEATURE_PATH).withColumn("label", F.col("label").cast("double"))
for c in numeric_features:
    df = df.withColumn(c, F.col(c).cast("double"))
df = df.filter(
    F.col("label").isin(0.0, 1.0) & F.col("first_order_ts").isNotNull()
)
df = df.withColumn("_purchase_epoch", F.unix_timestamp("first_order_ts"))
df = df.filter(F.col("_purchase_epoch").isNotNull())

cutoff = df.select(
    F.expr("percentile_approx(_purchase_epoch, 0.8, 10000)").alias("cutoff")
).first()["cutoff"]
train = df.filter(F.col("_purchase_epoch") < cutoff)
test = df.filter(F.col("_purchase_epoch") >= cutoff)

positive = train.filter(F.col("label") == 1.0).count()
negative = train.filter(F.col("label") == 0.0).count()
if positive == 0 or negative == 0:
    raise ValueError("Training data must contain both labels.")
total = positive + negative
positive_weight = total / (2.0 * positive)
negative_weight = total / (2.0 * negative)
train = train.withColumn(
    "classWeight",
    F.when(F.col("label") == 1.0, F.lit(positive_weight))
     .otherwise(F.lit(negative_weight))
)

print("Training records:", train.count())
print("Testing records:", test.count())
print("Continuing with Naive Bayes and the optional Linear Regression experiment.")

def build_stages(for_nb=False):
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

    if for_nb:
        # Multinomial Naive Bayes requires every input value to be >= 0.
        # Clip negative imputed numeric values to zero for this model only.
        # Other models and the source feature dataset are unchanged.
        nb_numeric = [c + "_nb_nonnegative" for c in numeric_features]
        selections = ["*"]
        for source, target in zip(imputed, nb_numeric):
            selections.append(
                f"greatest(CAST(`{source}` AS DOUBLE), 0.0) AS `{target}`"
            )
        stages.append(SQLTransformer(
            statement="SELECT " + ", ".join(selections) + " FROM __THIS__"
        ))
        assembler_inputs = nb_numeric + encoded
        output_col = "nb_features"
    else:
        assembler_inputs = imputed + encoded
        output_col = "features"

    stages.append(VectorAssembler(
        inputCols=assembler_inputs,
        outputCol=output_col,
        handleInvalid="keep"
    ))
    return stages

print("\n" + "=" * 70)
print("TRAINING NAIVE BAYES")
print("=" * 70)
nb = NaiveBayes(
    featuresCol="nb_features",
    labelCol="label",
    modelType="multinomial",
    smoothing=1.0
)
nb_pipeline = Pipeline(stages=build_stages(for_nb=True) + [nb])
nb_model = nb_pipeline.fit(train)
nb_predictions = nb_model.transform(test)

roc = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
).evaluate(nb_predictions)
pr = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
).evaluate(nb_predictions)

counts = nb_predictions.agg(
    F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("tp"),
    F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("tn"),
    F.sum(F.when((F.col("label") == 0.0) & (F.col("prediction") == 1.0), 1).otherwise(0)).alias("fp"),
    F.sum(F.when((F.col("label") == 1.0) & (F.col("prediction") == 0.0), 1).otherwise(0)).alias("fn")
).first()
tp, tn, fp, fn = [int(counts[k] or 0) for k in ("tp", "tn", "fp", "fn")]
precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
accuracy = (tp + tn) / (tp + tn + fp + fn)

nb_result = {
    "model": "Naive Bayes",
    "roc_auc": float(roc), "pr_auc": float(pr),
    "precision": float(precision), "recall": float(recall),
    "f1": float(f1), "accuracy": float(accuracy),
    "tp": tp, "tn": tn, "fp": fp, "fn": fn
}
print("Naive Bayes metrics:")
for key in ("roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"):
    print(f"{key}: {nb_result[key]:.5f}")
print(f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

nb_path = MODEL_BASE + "/repeat_purchase_v2_naive_bayes"
nb_model.write().overwrite().save(nb_path)
print("Saved Naive Bayes model to:", nb_path)

print("\n" + "=" * 70)
print("OPTIONAL LINEAR REGRESSION EXPERIMENT")
print("=" * 70)
lr = LinearRegression(
    featuresCol="features", labelCol="label", weightCol="classWeight",
    maxIter=50, regParam=0.01
)
lr_pipeline = Pipeline(stages=build_stages(for_nb=False) + [lr])
lr_model = lr_pipeline.fit(train)
lr_predictions = lr_model.transform(test)
regression_metrics = {}
for metric in ("rmse", "mae", "r2"):
    value = RegressionEvaluator(
        labelCol="label", predictionCol="prediction", metricName=metric
    ).evaluate(lr_predictions)
    regression_metrics[metric] = float(value)
    print(f"Linear Regression {metric.upper()}: {value:.5f}")

lr_path = MODEL_BASE + "/repeat_purchase_v2_linear_regression_experiment"
lr_model.write().overwrite().save(lr_path)
print("Saved Linear Regression experiment to:", lr_path)

# Preserve the five successful metrics from the first run and append the
# newly calculated Naive Bayes metrics. These are the default-threshold
# test-set metrics printed by the first run.
previous_results = [
    {"model":"Logistic Regression","roc_auc":0.5839855,"pr_auc":0.0356170,"precision":0.0304614,"recall":0.5184049,"f1":0.0575417,"accuracy":0.5987243,"tp":169,"tn":8091,"fp":5379,"fn":157},
    {"model":"Linear SVM","roc_auc":0.57695,"pr_auc":0.03677,"precision":0.02721,"recall":0.57669,"f1":0.05196,"accuracy":0.50275,"tp":188,"tn":6748,"fp":6722,"fn":138},
    {"model":"Decision Tree","roc_auc":0.47924,"pr_auc":0.02200,"precision":0.02803,"recall":0.49080,"f1":0.05303,"accuracy":0.58582,"tp":160,"tn":7922,"fp":5548,"fn":166},
    {"model":"Random Forest","roc_auc":0.55374,"pr_auc":0.03126,"precision":0.02854,"recall":0.37117,"f1":0.05300,"accuracy":0.68658,"tp":121,"tn":9351,"fp":4119,"fn":205},
    {"model":"Gradient-Boosted Trees","roc_auc":0.55647,"pr_auc":0.03043,"precision":0.02904,"recall":0.43252,"f1":0.05443,"accuracy":0.64490,"tp":141,"tn":8756,"fp":4714,"fn":185},
]
all_results = previous_results + [nb_result]
comparison = spark.createDataFrame(all_results)
print("\nSix-classifier comparison (sorted by PR-AUC):")
comparison.orderBy(F.desc("pr_auc")).show(truncate=False)
comparison.coalesce(1).write.mode("overwrite").option("header", True).csv(METRICS_PATH)
print("Saved six-classifier comparison to:", METRICS_PATH)
print("Linear Regression is a separate regression experiment, not a seventh classifier.")
print("CONTINUATION COMPLETED.")
spark.stop()
