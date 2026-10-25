# Brazilian E-Commerce Data Lakehouse

## 1. Project Overview

This project builds a data lakehouse using the Brazilian E-Commerce Public Dataset by Olist.

It uses distributed storage and processing technologies to organize e-commerce data and support analytical workloads. The project also includes a machine learning workflow to investigate whether customers are likely to make another purchase.

## 2. Technologies Used

- Python
- Apache Hadoop 3.3.6
- Hadoop Distributed File System (HDFS)
- Apache Spark 3.5.1
- YARN
- Docker and Docker Compose
- PySpark MLlib
- Git and GitHub

## 3. Data Lakehouse Architecture

The project organizes data into logical processing layers:

```text
Olist CSV Datasets
        |
        v
   Data Ingestion
        |
        v
   Raw Data Layer
        |
        v
Data Cleaning and Transformation
        |
        v
  Cleaned Data Layer
        |
        v
Data Curation and Analytics
        |
        v
Curated and Analytics Layers
        |
        v
   Analytical Outputs
```

HDFS provides distributed storage, while Apache Spark supports data processing. Docker Compose manages the containerized environment.

## 4. Repeat-Purchase Prediction

The machine learning component investigates whether customers are likely to make another purchase based on information about their first order.

The workflow includes:
1. Feature engineering.
2. Chronological training and testing split.
3. Handling class imbalance using class weights where supported.
4. Training and evaluating classification models.
5. Comparing models using multiple evaluation metrics.
6. Selecting a candidate model using Precision-Recall AUC (PR-AUC).

### Classification Models

Six classification algorithms were evaluated:
- Logistic Regression
- Linear Support Vector Machine (Linear SVM)
- Decision Tree
- Random Forest
- Gradient-Boosted Trees
- Naive Bayes

A Linear Regression experiment was also trained separately. Its regression metrics are not directly comparable with the classification metrics.

## 5. Model Comparison Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1-score | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.5840 | 0.0356 | 0.0305 | 0.5184 | 0.0575 | 0.5987 |
| Linear SVM | 0.5770 | 0.0368 | 0.0272 | 0.5767 | 0.0520 | 0.5028 |
| Decision Tree | 0.4792 | 0.0220 | 0.0280 | 0.4908 | 0.0530 | 0.5858 |
| Random Forest | 0.5537 | 0.0313 | 0.0285 | 0.3712 | 0.0530 | 0.6866 |
| Gradient-Boosted Trees | 0.5565 | 0.0304 | 0.0290 | 0.4325 | 0.0544 | 0.6449 |
| Naive Bayes | 0.5102 | 0.0242 | 0.0241 | 0.4325 | 0.0457 | 0.5734 |

### Candidate Model Selection

Linear SVM was selected because it achieved the highest PR-AUC, approximately 0.0368, among the six evaluated classifiers.

Logistic Regression achieved the highest ROC-AUC and F1-score. Random Forest achieved the highest accuracy, but accuracy alone is not a suitable selection criterion for this imbalanced dataset.

### Limitations

The test set contains many more non-repeat customers than repeat customers. Low precision and modest PR-AUC indicate that the models have limited predictive performance.

Linear SVM should be treated as a candidate for further improvement, not as a highly accurate production-ready model.

Some features describe the first-order experience, including delivery, review, and order-status information. Therefore, the prediction should be interpreted as occurring after the first-order experience, rather than strictly at checkout.

## 6. Evaluation Metrics

- **ROC-AUC:** Measures how well the model ranks positive examples above negative examples.
- **PR-AUC:** Summarizes the precision-recall trade-off and is useful for imbalanced datasets.
- **Precision:** The proportion of predicted repeat customers who actually repeat.
- **Recall:** The proportion of actual repeat customers identified by the model.
- **F1-score:** The harmonic mean of precision and recall.
- **Accuracy:** The proportion of all predictions that are correct.

## 7. Project Structure

```text
olist-lakehouse/
├── docker/
├── docs/
├── benchmarks/
├── sql/
├── src/
│   ├── 07_features.py
│   ├── 08_train.py
│   ├── 09_train_six_models.py
│   ├── 10_compare_models.py
│   └── 10_finish_naive_bayes.py
├── results/
│   └── model_comparison.csv
├── data/
└── README.md
```

## 8. Results Location

The classification comparison is available in:

`results/model_comparison.csv`

The training scripts save model artifacts and evaluation outputs to HDFS in the configured project environment.

## 9. Future Improvements

- Improve feature selection and engineering.
- Evaluate alternative methods for handling class imbalance.
- Tune classification thresholds using validation data.
- Perform systematic hyperparameter tuning.
- Compare against simple baseline models.
- Improve precision while retaining useful recall.
- Validate the final candidate on additional time periods.
- Review feature availability to prevent information leakage.

## 10. Repository

Source code and project documentation are maintained in the GitHub repository:

https://github.com/Bhavana666coder/olist-lakehouse
