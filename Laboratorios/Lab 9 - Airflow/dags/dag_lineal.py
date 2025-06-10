from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
from hiring_functions import create_folders, split_data, preprocess_and_train, predict, gradio_interface

# Ruta base donde se crearán las carpetas de ejecución
BASE_PATH = "/Users/sebastianacunau/airflow/ml_runs_lab9"

default_args = {
    'owner': 'seba',
    'retries': 1,
}

with DAG(
    dag_id="hiring_linear_pipeline",
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    description="Pipeline simple para predecir contratación de personas",
) as dag:

    # Tarea 1: Iniciar del DAG
    inicio = EmptyOperator(task_id="start_pipeline")

    # Tarea 2: Crear carpetas para la ejecución
    task_create_folders = PythonOperator(
        task_id="create_folders",
        python_callable=create_folders,
        provide_context=True,
    )

    # Tarea 3: Descargar el dataset data_1.csv con BashOperator
    task_download_dataset = BashOperator(
        task_id='download_data',
        bash_command="curl -o " 
                    f"{ ti.xcom_pull(task_ids='crear_folders') }/raw/data_1.csv "
                    "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    )

    # Tarea 4: Separar los datos en train/test
    task_split_data = PythonOperator(
        task_id="split_data",
        python_callable=split_data,
        provide_context=True,
    )

    # Tarea 5: Preprocesar los datos y entrenar el modelo
    task_preprocess_and_train = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train,
        provide_context=True,
    )

    # Tarea 6: Lanzar la interfaz de Gradio
    task_gradio_interface = PythonOperator(
        task_id="launch_gradio_interface",
        python_callable=gradio_interface,
        provide_context=True,
    )

    # Definir el orden de ejecución
    inicio >> task_create_folders >> task_download_dataset >> task_split_data >> task_preprocess_and_train >> task_gradio_interface
