cd ~/olist-lakehouse

cat > README.md <<'EOF'
# Brazilian E-Commerce Data Lakehouse

## Project Overview

This project implements a data lakehouse pipeline using the Brazilian E-Commerce Public Dataset by Olist. It uses Apache Hadoop HDFS for storage, Apache Spark for data processing, and Docker Compose to manage the environment.

The pipeline ingests raw CSV datasets, cleans and transforms data, creates a curated dataset, performs data quality checks, and generates analytical outputs.

## Technologies Used

- Python
- Apache Hadoop 3.3.6
- HDFS
- Apache Spark 3.5.1
- YARN
- Docker
- Docker Compose
- Pandas
- Parquet
- Git and GitHub

## Architecture

```text
Olist CSV Datasets
        |
        v
   Data Ingestion
        |
        v
Raw Layer: /datalake/raw/olist
        |
        v
   Data Cleaning
        |
        v
Cleaned Layer: /datalake/cleaned
        |
        v
   Data Curation
        |
        v
Curated Layer: /datalake/curated/order_analytics
        |
        v
 Analytics Processing
        |
        v
Analytics Layer: /datalake/analytics
        |
        v
Verification
