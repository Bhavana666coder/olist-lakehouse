#!/usr/bin/env python3

"""Ingest the Olist CSV datasets into HDFS."""

import subprocess
import sys
from pathlib import Path

LOCAL_DATA_DIR = Path("/app/data")
HDFS_RAW_DIR = "/datalake/raw/olist"

DATASETS = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def run_command(command):
    """Run a command and stop if it fails."""
    subprocess.run(command, check=True)


def main():
    if not LOCAL_DATA_DIR.is_dir():
        sys.exit(f"Data directory not found: {LOCAL_DATA_DIR}")

    # Check that every expected dataset exists before uploading anything.
    missing = [
        filename
        for filename in DATASETS
        if not (LOCAL_DATA_DIR / filename).is_file()
    ]

    if missing:
        sys.exit(f"Missing dataset files: {missing}")

    # Create the HDFS destination if it doesn't exist.
    run_command([
        "hdfs", "dfs", "-mkdir", "-p", HDFS_RAW_DIR
    ])

    # Upload only files that are not already present in HDFS.
    for filename in DATASETS:
        hdfs_path = f"{HDFS_RAW_DIR}/{filename}"

        result = subprocess.run(
            ["hdfs", "dfs", "-test", "-e", hdfs_path],
            check=False,
        )

        if result.returncode == 0:
            print(f"Already exists; skipping: {filename}")
            continue

        run_command([
            "hdfs", "dfs", "-put",
            str(LOCAL_DATA_DIR / filename),
            hdfs_path,
        ])

        print(f"Uploaded: {filename}")

    print("Raw ingestion completed successfully.")


if __name__ == "__main__":
    main()
