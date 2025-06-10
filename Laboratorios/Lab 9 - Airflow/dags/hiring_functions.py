import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import joblib
import gradio as gr


def create_folders(base_path="."):
    """
    Crea una carpeta con la fecha de hoy (YYYYMMDD_HHMMSS) y subcarpetas 'raw', 'splits' y 'models'.
    Retorna la ruta absoluta de la carpeta principal creada.
    """
    fecha = datetime.now(tz='America/Santiago').strftime("%Y%m%d_%H%M%S")
    main_folder = os.path.join(base_path, fecha)
    os.makedirs(main_folder, exist_ok=True)
    for sub in ["raw", "splits", "models"]:
        os.makedirs(os.path.join(main_folder, sub), exist_ok=True)
    return main_folder

def download_data(main_folder):
    """
    Descarga el archivo data_1.csv desde la URL proporcionada y lo guarda en main_folder/raw.
    """
    url = "https://gitlab.com/eduardomoyab/laboratorio-13/-/raw/main/files/data_1.csv"
    raw_path = os.path.join(main_folder, "raw", "data_1.csv")
    
    if not os.path.exists(raw_path):
        df = pd.read_csv(url)
        df.to_csv(raw_path, index=False)
        print(f"Archivo descargado y guardado en: {raw_path}")
    else:
        print(f"El archivo ya existe en: {raw_path}")

def split_data(main_folder):
    """
    Lee el archivo data_1.csv desde main_folder/raw,
    divide los datos en train/test (80/20, stratify=HiringDecision, random_state=13),
    y guarda ambos datasets en main_folder/splits.
    """
    raw_path = os.path.join(main_folder, "raw", "data_1.csv")
    df = pd.read_csv(raw_path)
    
    X = df.drop(columns=["HiringDecision"])
    y = df["HiringDecision"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=13
    )
    
    train_df = X_train.copy()
    train_df["HiringDecision"] = y_train
    test_df = X_test.copy()
    test_df["HiringDecision"] = y_test

    splits_path = os.path.join(main_folder, "splits")
    train_df.to_csv(os.path.join(splits_path, "train.csv"), index=False)
    test_df.to_csv(os.path.join(splits_path, "test.csv"), index=False)

def preprocess_and_train(main_folder):
    """
    Lee train.csv y test.csv de main_folder/splits, crea un pipeline con ColumnTransformer y RandomForest,
    entrena el modelo y guarda el pipeline entrenado en main_folder/models/model.joblib
    """
    splits_path = os.path.join(main_folder, "splits")
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))

    X_train = train_df.drop(columns=["HiringDecision"])
    y_train = train_df["HiringDecision"]
    X_test = test_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]

    pipeline = Pipeline([
        ("classifier", RandomForestClassifier(random_state=13))
    ])

    pipeline.fit(X_train, y_train)
    
    # Guardar el pipeline entrenado
    models_path = os.path.join(main_folder, "models")
    model_path = os.path.join(models_path, "model.joblib")
    joblib.dump(pipeline, model_path)

    # Calcular métricas en el conjunto de prueba
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print(f"Accuracy en test: {accuracy:.4f}")
    print(f"F1-score (clase positiva=1) en test: {f1:.4f}")

    return model_path

def get_latest_model_path(base_path="."):
    """
    Busca la carpeta más reciente en base_path y retorna el path al model.joblib dentro de models/
    """
    folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    if not folders:
        raise FileNotFoundError("No se encontraron carpetas de ejecuciones anteriores.")
    folders.sort(key=lambda x: os.path.getmtime(os.path.join(base_path, x)), reverse=True)
    latest_folder = os.path.join(base_path, folders[0])
    model_path = os.path.join(latest_folder, "models", "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró modelo en {model_path}")
    return model_path


def predict(file,model_path):

    pipeline = joblib.load(model_path)
    input_data = pd.read_json(file)
    predictions = pipeline.predict(input_data)
    print(f'La prediccion es: {predictions}')
    labels = ["No contratado" if pred == 0 else "Contratado" for pred in predictions]

    return {'Predicción': labels[0]}

def gradio_interface(base_path="."):

    model_path = get_latest_model_path(base_path) #Completar con la ruta del modelo entrenado

    interface = gr.Interface(
        fn=lambda file: predict(file, model_path),
        inputs=gr.File(label="Sube un archivo JSON"),
        outputs="json",
        title="Hiring Decision Prediction",
        description="Sube un archivo JSON con las características de entrada para predecir si Vale será contratada o no."
    )
    interface.launch(share=True)
