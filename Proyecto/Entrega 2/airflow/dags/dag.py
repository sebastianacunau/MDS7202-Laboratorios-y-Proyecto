from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from datetime import datetime, timedelta

from modular_functions import (
    retrieve_data,
    standardize_and_prepare_data,
    detect_data_drift,
    choose_drift_branch,
    split_data,
    optimize_model,
    interpret_model,
    evaluate_model,
    generate_predictions
)

# Definición de argumentos por defecto del DAG
default_args = {
    'owner': 'deep_drinkers',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'start_date': datetime(2025, 6, 29),
}

with DAG(
    dag_id = "productive_pipeline_for_predictions_with_data_drift_detection",
    default_args=default_args,
    schedule_interval="@daily",  # Se ejecuta diariamente
    catchup=False, 
    description="Pipeline productivo para predicciones con detección de drift de datos",
) as dag:
    
    # Tarea 1: Iniciar el DAG
    start_pipeline = EmptyOperator(task_id="start_pipeline")

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
    task_detect_drift = PythonOperator(
        task_id="detect_data_drift",
        python_callable=detect_data_drift,
        provide_context=True,
    )

    # Tarea 5: Branching para decidir si se reentrena el modelo o no
    task_branch = BranchPythonOperator(
        task_id="branching",
        python_callable=choose_drift_branch,
        provide_context=True,
    )

    # Tarea 5.a: Si no se detecta drift, saltar el entrenamiento
    task_skip_training = EmptyOperator(task_id="skip_training")

    # Tarea 5.b.1: Dividir los datos en entrenamiento y prueba
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data, 
        provide_context=True,
    )

    # Tarea 5.b.2: Optimizar el modelo
    task_optimize_model = PythonOperator(
        task_id="optimize_model",
        python_callable=optimize_model,
        provide_context=True,
    )

    # Tarea 6: Evaluar el modelo
    task_evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        provide_context=True,
    )

    # Tarea 7: Interpretar el modelo
    task_interpret_model = PythonOperator(
        task_id="interpret_model",
        python_callable=interpret_model,
        provide_context=True,
    )

    # Tarea 8: Generar predicciones
    task_generate_predictions = PythonOperator(
    task_id="generate_predictions",
    python_callable=generate_predictions,
    provide_context=True,
    )

    # Tarea 9: Finalizar el DAG
    end_pipeline = EmptyOperator(task_id="end_pipeline")

    # Definir el flujo
    start_pipeline >> task_retrieve_data >> task_standardize_and_prepare >> task_detect_drift >> task_branch
    
    task_branch >> task_skip_training >> task_generate_predictions >> task_evaluate_model >> task_interpret_model >> end_pipeline

    task_branch >> task_split_data >> task_optimize_model >> task_generate_predictions
    task_generate_predictions >> task_evaluate_model >> task_interpret_model >> end_pipeline