import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr


def create_folders(**kwargs):
    """
    Crea una carpeta con la fecha de hoy (YYYYMMDD_HHMMSS) y subcarpetas 'raw', 'splits' y 'models'.
    Retorna la ruta absoluta de la carpeta principal creada.
    """
    ti = kwargs['ti']

    # Obtener la fecha desde el contexto del DAG
    fecha = kwargs['dag_run'].conf.get('start_date', datetime.now().strftime("%Y%m%d_%H%M%S"))

    # Crear la carpeta principal con el nombre de la fecha
    main_folder = os.path.join(ti.xcom_pull(key="base_path"), fecha)
    os.makedirs(main_folder, exist_ok=True)
    
    # Crear subcarpetas
    for sub in ["raw", "splits", "models"]:
        os.makedirs(os.path.join(main_folder, sub), exist_ok=True)
    
    # Guardar la ruta completa de la carpeta principal en XCom
    ti.xcom_push(key='main_folder', value=main_folder)

def split_data(**kwargs):
    """
    Lee el archivo data_1.csv desde main_folder/raw,
    divide los datos en train/test (80/20, stratify=HiringDecision, random_state=13),
    y guarda ambos datasets en main_folder/splits.
    """
    ti = kwargs['ti']

    # Obtener la ruta de la carpeta principal desde XCom
    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')

    # Leer el archivo data_1.csv desde la carpeta raw
    raw_path = os.path.join(main_folder, "raw", "data_1.csv")
    df = pd.read_csv(raw_path)
    
    # Dividir los datos en entrenamiento y prueba
    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=13)
    
    train_df = X_train.copy()
    train_df["HiringDecision"] = y_train
    test_df = X_test.copy()
    test_df["HiringDecision"] = y_test
    
    # Guardar los datasets en la carpeta splits
    splits_path = os.path.join(main_folder, "splits")
    train_df.to_csv(os.path.join(splits_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(splits_path, "test.csv"), index=False)

def preprocess_and_train(**kwargs):
    """
    Lee train.csv y test.csv de main_folder/splits, crea un pipeline con ColumnTransformer y RandomForest,
    entrena el modelo y guarda el pipeline entrenado en main_folder/models/model.joblib
    """
    ti = kwargs['ti']

    # Obtener la ruta de la carpeta principal desde XCom
    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')
    splits_path = os.path.join(main_folder, "splits")

    # Leer los datasets de entrenamiento y prueba
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))

    X_train = train_df.drop(columns=["HiringDecision"])
    y_train = train_df["HiringDecision"]
    X_test = test_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]

    # Crear un pipeline (no se realiza preprocesamiento en este ejemplo, pues RF no lo necesita, pero se puede agregar)
    pipeline = Pipeline([
        ("classifier", RandomForestClassifier(random_state=13))
    ])
    pipeline.fit(X_train, y_train)

    # Calcular métricas en el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Accuracy en test: {accuracy:.4f}")
    print(f"F1-score (clase positiva=1) en test: {f1:.4f}")

    # Guardar el pipeline entrenado
    models_path = os.path.join(main_folder, "models")
    model_path = os.path.join(models_path, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Guardar el path del modelo en XCom para su uso posterior
    ti.xcom_push(key='model_path', value=model_path)

def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}

def gradio_interface(**kwargs):

    ti = kwargs['ti']
    model_path = ti.xcom_pull(task_ids='preprocess_and_train', key='model_path')
    if not model_path:
        raise ValueError("El modelo no ha sido entrenado o no se encontró la ruta del modelo.")

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
