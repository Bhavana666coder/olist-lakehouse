import csv
import json
import time
import argparse
from pathlib import Path
from kafka import KafkaProducer
from kafka.errors import KafkaError


def load_orders(orders_file):
    orders = {}

    with open(orders_file, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            orders[row["order_id"]] = {
                "customer_id": row["customer_id"],
                "order_status": row["order_status"],
                "order_purchase_timestamp": row["order_purchase_timestamp"],
            }

    return orders


def main():
    parser = argparse.ArgumentParser(
        description="Publish Olist order-item events to Kafka"
    )
    parser.add_argument("--limit", type=int, default=100,
                        help="Maximum number of events to publish (default: 100)")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="Delay in seconds between events (default: 0.1)")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent.parent
    orders_file = project_dir / "data" / "olist_orders_dataset.csv"
    items_file = project_dir / "data" / "olist_order_items_dataset.csv"

    print("Loading Olist orders...")
    orders = load_orders(orders_file)
    print(f"Loaded {len(orders):,} orders.")

    producer = KafkaProducer(
        bootstrap_servers=["localhost:9092"],
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
        acks="all",
        retries=3,
    )

    sent = 0

    try:
        with open(items_file, "r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file)

            for item in reader:
                order = orders.get(item["order_id"])

                if order is None:
                    continue

                event = {
                    "order_id": item["order_id"],
                    "order_item_id": int(item["order_item_id"]),
                    "customer_id": order["customer_id"],
                    "order_status": order["order_status"],
                    "order_purchase_timestamp": order[
                        "order_purchase_timestamp"
                    ],
                    "product_id": item["product_id"],
                    "seller_id": item["seller_id"],
                    "price": float(item["price"]),
                    "freight_value": float(item["freight_value"]),
                }

                producer.send("olist_orders", value=event).get(timeout=30)
                sent += 1

                if sent % 25 == 0:
                    print(f"Published {sent} events...")

                if args.delay > 0:
                    time.sleep(args.delay)

                if sent >= args.limit:
                    break

        producer.flush()
        print(f"\nSuccessfully published {sent} events to 'olist_orders'.")

    except (KafkaError, OSError, ValueError) as error:
        print(f"Error while publishing events: {error}")
        raise

    finally:
        producer.close()


if __name__ == "__main__":
    main()
