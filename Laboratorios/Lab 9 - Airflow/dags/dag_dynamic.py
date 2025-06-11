from airflow import DAG

from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from airflow.operators.branch import BranchPythonOperator
from airflow.utils.dates import days_ago

from datetime import datetime, timedelta

from hiring_functions import create_folders, choose_data_branch, load_and_merge, split_data, train_model, evaluate_models

# Configuración del DAG
default_args = {
    'owner': 'seba',
    'retries': 1,
}

with DAG(
    dag_id="hiring_dynamic_pipeline_with_backfill",
    default_args=default_args,
    start_date=datetime(2024, 10, 1),
    schedule_interval="5 15 1 * *",  # Se ejecuta el día 5 de cada mes a las 15:00 UTC
    catchup=True,  # Habilita backfill
    description="Pipeline dinámico con evaluación y modelos paralelos",
) as dag:

    # Tarea 1: Iniciar el DAG
    inicio = EmptyOperator(task_id="start_pipeline")

    # Tarea 2: Crear carpetas para la ejecución
    task_create_folders = PythonOperator(
        task_id="crear_folders",
        python_callable=create_folders,
        provide_context=True,
    )

    # Tarea 3: Branching para seleccionar los archivos a descargar
    task_branching = BranchPythonOperator(
        task_id="choose_data_branch",
        python_callable=choose_data_branch,
        provide_context=True,
        dag=dag,
    )

    # Tarea 4.a: Descargar dataset 1
    task_download_dataset_1 = BashOperator(
        task_id='download_dataset_1',
        bash_command=(
            "curl -o {{ ti.xcom_pull(task_ids='crear_folders', key='main_folder') }}/raw/data_1.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
        ),
    )

    # Tarea 4.b: Descargar dataset 2 según la fecha
    task_download_dataset_2 = BashOperator(
        task_id='download_dataset_2',
        bash_command=(
            "curl -o {{ ti.xcom_pull(task_ids='crear_folders', key='main_folder') }}/raw/data_2.csv "
            "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_2.csv"
        ),
    )

    # Tarea 5: Concatenar datasets disponibles
    task_load_and_merge = PythonOperator(
        task_id="load_and_merge",
        python_callable=load_and_merge,
        provide_context=True,
        trigger_rule='one_success',  # Se ejecuta si al menos una de las tareas de descarga tiene éxito
    )

    # Tarea 6: Split de datos en train/test
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        provide_context=True,
    )

    # Tarea 7: Entrenamiento de 3 modelos en paralelo
    task_train_rf = PythonOperator(
        task_id="train_random_forest",
        python_callable=train_model,
        op_args=['RandomForest', 'model_rf.joblib'],
        provide_context=True,
    )

    task_train_dt = PythonOperator(
        task_id="train_decision_tree",
        python_callable=train_model,
        op_args=['DecisionTree', 'model_dt.joblib'],
        provide_context=True,
    )

    task_train_lr = PythonOperator(
        task_id="train_logistic_regression",
        python_callable=train_model,
        op_args=['ModelType3', 'model_lr.joblib'],
        provide_context=True,
    )

    # Tarea 8: Evaluación de los modelos
    evaluate_models_task = PythonOperator(
        task_id="evaluate_models",
        python_callable=evaluate_models,
        provide_context=True,
        trigger_rule='all_success',  # Se ejecuta si todas las tareas de entrenamiento tienen éxito
    )

    # Definir el flujo de trabajo del DAG
    inicio >> task_create_folders >> task_branching

    # Tareas de descarga según la fecha
    task_branching >> [task_download_dataset_1, task_download_dataset_2]

    # Tareas de concatenación y split
    [task_download_dataset_1, task_download_dataset_2] >> task_load_and_merge >> task_split_data

    # Entrenamiento de modelos en paralelo
    task_split_data >> [task_train_rf, task_train_dt, task_train_lr]

    # Evaluación de los modelos entrenados
    [task_train_rf, task_train_dt, task_train_lr] >> evaluate_models_task


