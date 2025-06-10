from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from datetime import datetime
from hiring_functions import create_folders, download_data, split_data, preprocess_and_train, gradio_interface

# Ruta base donde se crearán las carpetas de ejecución
BASE_PATH = "/Users/sebastianacunau/airflow/ml_runs"  # Cambia esto según tu estructura

default_args = {
    'owner': 'seba',
}

with DAG(
    dag_id="hiring_linear_pipeline",
    start_date=datetime(2024, 10, 1),
    schedule_interval=None,
    catchup=False,
    default_args=default_args,
    description="Pipeline simple para predecir contratación de personas",
    tags=["ML", "hiring", "demo"],
) as dag:

    inicio = EmptyOperator(task_id="inicio")

    def crear_folders_callable(**kwargs):
        # Creamos la carpeta principal y retornamos el path
        main_folder = create_folders(base_path=BASE_PATH)
        # Guardamos el path usando XCom para pasarlo a otras tareas
        kwargs['ti'].xcom_push(key='main_folder', value=main_folder)

    crear_folders = PythonOperator(
        task_id="crear_folders",
        python_callable=crear_folders_callable,
        provide_context=True,
    )

    def download_data_callable(**kwargs):
        main_folder = kwargs['ti'].xcom_pull(key='main_folder', task_ids='crear_folders')
        download_data(main_folder)  # Asume que tu función sabe a dónde guardar

    descarga_datos = PythonOperator(
        task_id="descarga_datos",
        python_callable=download_data_callable,
        provide_context=True,
    )

    def split_data_callable(**kwargs):
        main_folder = kwargs['ti'].xcom_pull(key='main_folder', task_ids='crear_folders')
        split_data(main_folder)

    split = PythonOperator(
        task_id="split_data",
        python_callable=split_data_callable,
        provide_context=True,
    )

    def preprocess_and_train_callable(**kwargs):
        main_folder = kwargs['ti'].xcom_pull(key='main_folder', task_ids='crear_folders')
        preprocess_and_train(main_folder)

    entrenar = PythonOperator(
        task_id="preprocess_and_train",
        python_callable=preprocess_and_train_callable,
        provide_context=True,
    )

    def gradio_interface_callable(**kwargs):
        main_folder = kwargs['ti'].xcom_pull(key='main_folder', task_ids='crear_folders')
        gradio_interface(main_folder)

    lanzar_gradio = PythonOperator(
        task_id="lanzar_gradio_interface",
        python_callable=gradio_interface_callable,
        provide_context=True,
    )

    # Definir el orden de ejecución
    inicio >> crear_folders >> descarga_datos >> split >> entrenar >> lanzar_gradio
