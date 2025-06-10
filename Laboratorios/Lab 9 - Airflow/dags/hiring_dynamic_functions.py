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
    for sub in ["raw", "preprocessed", "splits", "models"]:
        os.makedirs(os.path.join(main_folder, sub), exist_ok=True)
    return main_folder

def loads_and_merge():
    """
    Carga los archivos data_1.csv y data_2.csv desde la carpeta 'raw',
    los une en un único DataFrame y lo guarda en 'preprocessed/merged_data.csv'.
    """
    raw_path = os.path.join("raw")
    df1 = pd.read_csv(os.path.join(raw_path, "data_1.csv"))
    df2 = pd.read_csv(os.path.join(raw_path, "data_2.csv"))
    
    merged_df = pd.concat([df1, df2], ignore_index=True)
    preprocessed_path = os.path.join("preprocessed")
    merged_df.to_csv(os.path.join(preprocessed_path, "merged_data.csv"), index=False)

def split_data(main_folder):
    """
    Lee el archivo merged_data.csv desde main_folder/preprocessed,
    divide los datos en train/test (80/20, stratify=HiringDecision, random_state=13),
    y guarda ambos datasets en main_folder/splits.
    """
    preprocessed_path = os.path.join(main_folder, "preprocessed", "merged_data.csv")
    df = pd.read_csv(preprocessed_path)
    
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

def train_model(model):
    pass