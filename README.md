# Olist Lakehouse: Big Data Processing and Analytics

## 1. Project Overview

Olist Lakehouse is a big data engineering project that processes Brazilian e-commerce data using Apache Hadoop, HDFS, Apache Spark, and PySpark.

The project implements a data pipeline that ingests raw CSV datasets, cleans and transforms the data, creates a curated analytical dataset, performs data-quality checks, and generates analytics for sales, orders, payments, and customer locations.

The project uses Docker to run the Hadoop and Spark environment.

## 2. Objectives

- Build a distributed-storage environment using Hadoop HDFS.
- Ingest raw e-commerce datasets into HDFS.
- Clean datasets and store the processed data in Parquet format.
- Create a curated dataset by combining order, customer, and payment information.
- Identify duplicate records and missing values.
- Verify the curated dataset and analytics outputs.
- Generate sales and order analytics using PySpark.
- Store analytical outputs in HDFS for further analysis.

## 3. Technologies Used

| Technology | Purpose |
|---|---|
| Apache Hadoop 3.3.6 | Distributed storage and Hadoop ecosystem |
| HDFS | Storage of raw, cleaned, curated, and analytical data |
| Apache Spark 3.5.1 | Distributed data processing |
| PySpark | Data cleaning, transformation, aggregation, and analytics |
| Docker | Containerized execution environment |
| Docker Compose | Management of Hadoop and Spark services |
| Python | Data pipeline implementation |
| Parquet | Columnar storage format for processed data |
| Git and GitHub | Version control and project hosting |

## 4. System Architecture

The pipeline follows these stages:

1. **Raw ingestion:** CSV datasets are ingested into HDFS.
2. **Data cleaning:** Raw datasets are processed to remove duplicate records and stored as Parquet files.
3. **Data curation:** Orders, customers, and payment information are combined to create an analytical dataset.
4. **Data-quality checks:** Processed datasets are examined for duplicate records and missing values.
5. **Curated-data verification:** The curated dataset is checked for its record count and schema.
6. **Analytics:** Sales, order-status, customer-city, and payment summaries are calculated.
7. **Analytics verification:** The generated analytical datasets are checked for successful creation and readable contents.

### Architecture Flow

```text
Brazilian E-Commerce CSV Dataset
              |
              v
       Raw Data Ingestion
              |
              v
       Hadoop HDFS Raw Layer
              |
              v
       PySpark Data Cleaning
              |
              v
      HDFS Cleaned Layer
         (Parquet Files)
              |
              v
       Data Curation
              |
              v
   Curated Order Analytics
              |
              v
       PySpark Analytics
              |
              v
       HDFS Analytics Layer
              |
              v
      Verification and Results
```

## 5. Project Structure

```text
olist-lakehouse/
├── docker/
│   ├── Dockerfile.hadoop
│   ├── Dockerfile.spark
│   ├── docker-compose.yml
│   ├── hadoop.env
│   └── hadoop-config/
│       ├── core-site.xml
│       ├── hdfs-site.xml
│       ├── mapred-site.xml
│       └── yarn-site.xml
├── src/
│   ├── 01_ingest_raw.py
│   ├── 02_clean.py
│   ├── 03_curate.py
│   ├── 04_verify_curated.py
│   ├── 05_analytics.py
│   ├── 06_verify_analytics.py
│   ├── dq_checks.py
│   └── test_yarn.py
├── sql/
├── docs/
├── benchmarks/
├── data/                 # Local dataset files; excluded from Git
├── .gitignore
└── README.md
```

## 6. Data Pipeline

### Stage 1: Raw Data Ingestion

Script: `src/01_ingest_raw.py`

The ingestion stage loads the source CSV datasets into HDFS under:

```text
/datalake/raw/olist
```

This provides a centralized storage layer for subsequent processing.

### Stage 2: Data Cleaning

Script: `src/02_clean.py`

The cleaning stage processes nine source datasets, removes duplicate records, and writes the cleaned data to HDFS in Parquet format.

Output location:

```text
/datalake/cleaned/
```

### Stage 3: Data Curation

Script: `src/03_curate.py`

The curation stage combines order information with customer information and aggregated payment data.

The curated dataset includes order details, customer attributes, purchase year and month, and total payment value.

Output location:

```text
/datalake/curated/order_analytics
```

The pipeline successfully produced **99,441 curated records**.

### Stage 4: Data-Quality Checks

Script: `src/dq_checks.py`

The data-quality stage examines the nine cleaned datasets for:

- Total record counts
- Column counts
- Duplicate rows
- Missing values by column

Missing values are reported for interpretation rather than automatically treating every missing value as an error. For example, some orders may not have delivery timestamps, and review comments may be absent.

### Stage 5: Curated-Data Verification

Script: `src/04_verify_curated.py`

This stage reads the curated Parquet dataset and verifies its schema and record count.

The completed verification reported **99,441 records**.

### Stage 6: Analytics

Script: `src/05_analytics.py`

The analytics stage generates five analytical datasets.

| Output | Description |
|---|---|
| `monthly_sales` | Total payment value and order count grouped by purchase year and month |
| `order_status` | Number of orders for each order status |
| `top_customer_cities` | Order counts grouped by customer city |
| `payment_summary` | Overall payment total, average payment per order, and order count |
| `yearly_sales` | Total payment value and order count grouped by purchase year |

Analytics output location:

```text
/datalake/analytics/
```

The generated analytical datasets are stored in Parquet format.

### Stage 7: Analytics Verification

Script: `src/06_verify_analytics.py`

The verification script reads all five analytical datasets from HDFS, displays their columns and record counts, and prints a completion message after processing them.

The completed verification reported the following output counts:

| Dataset | Rows |
|---|---:|
| `monthly_sales` | 25 |
| `order_status` | 8 |
| `top_customer_cities` | 4,119 |
| `payment_summary` | 1 |
| `yearly_sales` | 3 |

These counts confirm that the datasets were readable and contained records. They do not, by themselves, establish that every calculated business metric is correct.

## 7. Docker Environment

The project uses Docker Compose to run the following services:

| Service | Responsibility |
|---|---|
| NameNode | Manages HDFS filesystem metadata |
| DataNode | Stores HDFS data blocks |
| ResourceManager | Manages YARN cluster resources |
| NodeManager | Manages resources on a worker node |
| Spark client | Runs PySpark processing and analytics scripts |

The Docker Compose configuration is located at:

```text
docker/docker-compose.yml
```

The Hadoop configuration files are located in:

```text
docker/hadoop-config/
```

The environment uses HDFS replication factor 1, which is appropriate for this single-DataNode development setup but does not provide redundant storage.

## 8. Running the Project

The following commands are intended to be run in the Ubuntu VM terminal from the project directory.

### Prerequisites

- Docker
- Docker Compose
- The project source files
- The Brazilian e-commerce CSV dataset in the local `data/` directory

### Step 1: Start the environment

```bash
cd ~/olist-lakehouse

docker compose -f docker/docker-compose.yml up -d --build
```

Check the running containers:

```bash
docker ps
```

### Step 2: Run the pipeline scripts

The following command pattern runs a script inside the Spark client container:

```bash
docker exec spark-client bash -lc '
export HADOOP_CONF_DIR=/opt/hadoop/etc/hadoop
export PYSPARK_PYTHON=/usr/bin/python3

/opt/spark/bin/spark-submit \
  --master "local[2]" \
  --conf spark.pyspark.python=/usr/bin/python3 \
  /app/src/01_ingest_raw.py
'
```

Run the scripts in the following order:

1. `01_ingest_raw.py`
2. `02_clean.py`
3. `03_curate.py`
4. `dq_checks.py`
5. `04_verify_curated.py`
6. `05_analytics.py`
7. `06_verify_analytics.py`

To run another script, replace `01_ingest_raw.py` in the command with the desired script filename.

**Note:** These commands assume that the Docker containers have been built and started and that the mounted source files and Python environment are available. If the environment has been changed, verify the paths and Python configuration before running the scripts.

## 9. Verifying HDFS Outputs

To inspect the HDFS directories, run:

```bash
docker exec namenode /opt/hadoop/bin/hdfs dfs -ls /datalake/raw/olist
```

Check the cleaned data:

```bash
docker exec namenode /opt/hadoop/bin/hdfs dfs -ls /datalake/cleaned
```

Check the curated dataset:

```bash
docker exec namenode /opt/hadoop/bin/hdfs dfs -ls /datalake/curated/order_analytics
```

Check the analytics output:

```bash
docker exec namenode /opt/hadoop/bin/hdfs dfs -ls /datalake/analytics
```

The analytical output directories are:

```text
/datalake/analytics/monthly_sales
/datalake/analytics/order_status
/datalake/analytics/top_customer_cities
/datalake/analytics/payment_summary
/datalake/analytics/yearly_sales
```

## 10. Data Quality Observations

The data-quality checks identified missing values in some order, product, and review fields.

Examples include:

- Missing delivery timestamps in the orders dataset
- Missing product attributes
- Missing review comments and some review-related fields

These observations are reported for further analysis. Missing values may reflect the source data and should be interpreted according to the meaning of each field.

## 11. Limitations and Future Improvements

Potential improvements include:

- Implementing and integrating reusable schemas in `schemas.py`.
- Adding automated tests for the transformation and analytics logic.
- Defining explicit data-quality rules and thresholds.
- Validating calculated sales and payment metrics against independent totals.
- Adding more analytical queries and visualizations.
- Improving storage redundancy for a production deployment.
- Adding resource monitoring and performance benchmarks.
- Creating automated pipeline orchestration and scheduled execution.
- Documenting dataset provenance and licensing information.

## 12. Repository and Dataset Notes

The local dataset files, Python virtual environment, and Python build artifacts are excluded from version control.

Anyone reproducing the project must obtain the appropriate source dataset and place the CSV files in the expected local data directory before running ingestion.

The local execution logs are not required for the pipeline itself. Verification results can be regenerated by running the relevant scripts.

## 13. Conclusion

The Olist Lakehouse project demonstrates an end-to-end big data processing workflow using Hadoop HDFS, Docker, Apache Spark, and PySpark.

It includes raw data ingestion, data cleaning, data curation, data-quality checks, curated-data verification, analytical aggregation, and analytics verification. The completed pipeline produced 99,441 curated records and five readable analytical datasets stored in HDFS.

The project provides a foundation for further development of data engineering workflows and analytical processing.
