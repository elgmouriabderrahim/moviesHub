from datetime import datetime

from airflow import DAG
from airflow.providers.standard.operators.bash import BashOperator


PROJECT = "/opt/movieshub"

with DAG(
    dag_id="movieshub_pipeline",
    description="Extract, clean, engineer features, load MongoDB, and train ML artifacts",
    start_date=datetime(2026, 10, 1),
    schedule=None,
    catchup=False,
) as dag:
    extract = BashOperator(
        task_id="extract",
        bash_command=f"python {PROJECT}/src/extract.py",
    )
    clean = BashOperator(
        task_id="clean",
        bash_command=f"python {PROJECT}/src/clean.py",
    )
    features = BashOperator(
        task_id="features",
        bash_command=f"python {PROJECT}/src/build_features.py",
    )
    mongo = BashOperator(
        task_id="mongodb_upsert",
        bash_command=f"python {PROJECT}/src/load_mongodb.py",
    )
    ml = BashOperator(
        task_id="train_and_save_ml_artifacts",
        bash_command=f"python {PROJECT}/src/train_models.py",
    )

    extract >> clean >> features >> mongo >> ml
