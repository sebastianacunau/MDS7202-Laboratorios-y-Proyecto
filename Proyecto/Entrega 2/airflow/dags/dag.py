from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.utils.dates import days_ago

from datetime import datetime, timedelta

from modular_functions import retrieve_data, standardize_and_prepare_data, detect_data_drift, optimize_model, train_model, interpret_model, evaluate_model

default_args = {
    'owner': 'G2',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    dag_id = "productive_pipeline_for_predictions_with_data_drift_detection",
    default_args=default_args,
    schedule_interval="@daily",  # Se ejecuta diariamente
    catchup=True,  # Habilita backfill
    description="Pipeline productivo para predicciones con detección de drift de datos",
) as dag:
    
    # Tarea 1: Iniciar el DAG
    inicio = EmptyOperator(task_id="start_pipeline")

    # Tarea 2: Recuperar datos
    task_retrieve_data = PythonOperator(
        task_id="retrieve_data",
        python_callable=retrieve_data,
        provide_context=True,
    )

    # Tarea 3: Estandarizar y preparar los datos
    task_standardize_and_prepare = PythonOperator(
        task_id="standardize_and_prepare_data",
        python_callable=standardize_and_prepare_data,
        provide_context=True,
    )

    # Tarea 4: Detectar drift de datos
    task_detect_data_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
        provide_context=True,
    )

    # Tarea 5: Cargar y fusionar los datasets descargados
    task_load_and_merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        provide_context=True,
    )

    # Tarea 6: Dividir los datos en conjuntos de entrenamiento y prueba
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        provide_context=True,
    )

    # Tareas para entrenar modelos paralelos
    models = ['RandomForestClassifier', 'DecisionTreeClassifier', 'LogisticRegression']
    
    tasks_train_models = []
    for model in models:
        tasks_train_models.append(
            PythonOperator(
                task_id=f"train_{model.lower()}",
                python_callable=train_model,
                op_kwargs={'model_name': model},
                provide_context=True,