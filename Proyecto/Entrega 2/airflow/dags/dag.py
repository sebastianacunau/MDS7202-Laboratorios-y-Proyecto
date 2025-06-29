from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.utils.dates import days_ago

from datetime import datetime, timedelta

from modular_functions import retrieve_data, standardize_and_prepare_data, detect_data_drift, optimize_model, train_model, interpret_model, evaluate_model

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

    # Tarea 4: Branching para decidir si se reentrena el modelo o no
    branch = BranchPythonOperator(
        task_id="branching",
        python_callable=lambda ti: "retrain_model" if ti.xcom_pull(task_ids='detect_data_drift', key='drift_detected') else "skip_training",
        provide_context=True,
    )

    # Tarea 5.a: Si se detecta drift, reentrenar el modelo
    task_retrain_model = PythonOperator(
        task_id="retrain_model",
        python_callable=optimize_model,
        provide_context=True,
    )

    # Tarea 5.a.1: Dividir los datos en entrenamiento y prueba
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data, 
        provide_context=True,
    )

    # Tarea 5.a.2: Entrenar modelos paralelamente
    tasks_train_models = []
    model_types = ['RandomForest', 'DecisionTree', 'LogisticRegression']
    for model_type in model_types:
        task_train_model = PythonOperator(
            task_id=f"train_model_{model_type}",
            python_callable=train_model,
            op_args=[model_type, f"{model_type.lower()}_model.pkl"],
            provide_context=True,
        )
        tasks_train_models.append(task_train_model)

    # Configurar las tareas de entrenamiento en paralelo
    for i in range(len(tasks_train_models) - 1):
        tasks_train_models[i] >> tasks_train_models[i + 1]

    # Tarea 5.a.3: Optimizar el modelo
    task_optimize_model = PythonOperator(
        task_id="optimize_model",
        python_callable=optimize_model,
        provide_context=True,
    )

    # Tarea 5.b: Si no se detecta drift, saltar el entrenamiento
    task_skip_training = EmptyOperator(task_id="skip_training")

    # Tarea 6: Predecir con el modelo
    task_predict = PythonOperator(
        task_id="predict",
        python_callable=train_model,
        provide_context=True,
    )

    # Tarea 7: Evaluar el modelo
    task_evaluate_model = PythonOperator(
        task_id="evaluate_model",
        python_callable=evaluate_model,
        provide_context=True,
    )

    # Tarea 8: Interpretar el modelo
    task_interpret_model = PythonOperator(
        task_id="interpret_model",
        python_callable=interpret_model,
        provide_context=True,
    )

    # Tarea 9: Finalizar el DAG
    task_end = EmptyOperator(task_id="end_pipeline")

    # Definir el flujo
    inicio >> task_retrieve_data >> task_standardize_and_prepare >> task_detect_data_drift >> branch
    branch >> task_split_data >> tasks_train_models >> task_predict >> task_evaluate_model >> task_interpret_model >> task_end
    branch >> task_skip_training >> task_predict >> task_evaluate_model >> task_interpret_model >> task_end
