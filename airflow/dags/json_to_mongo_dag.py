from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
from pymongo import MongoClient
import json


def load_json_to_mongo():
    mongo_uri = "mongodb://mongo_user:mongo_pass@mongo:27017/airflow_db?authSource=admin"

    json_path = "/opt/airflow/data/BRONZE/orders.json"

    client = MongoClient(mongo_uri)
    db = client["airflow_db"]
    collection = db["orders"]

    with open(json_path, "r") as f:
        data = json.load(f)

    if isinstance(data, list):
        collection.insert_many(data)
        count = len(data)
    else:
        collection.insert_one(data)
        count = 1

    print(f"Inserted {count} documents into MongoDB")


with DAG(
    dag_id="json_to_mongo_dag",
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mongo", "json"],
) as dag:

    load_task = PythonOperator(
        task_id="load_json_into_mongo",
        python_callable=load_json_to_mongo,
    )
