import os
from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score
import joblib


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
    for sub in ["raw", "preprocessed", "splits", "models"]:
        os.makedirs(os.path.join(main_folder, sub), exist_ok=True)
    
    # Guardar la ruta completa de la carpeta principal en XCom
    ti.xcom_push(key='main_folder', value=main_folder)

def choose_data_branch(**kwargs):
    """
    Esta función de branching selecciona qué archivos descargar dependiendo de la fecha.
    Si la fecha es anterior a 1 de noviembre de 2024, descarga solo data_1.csv.
    De lo contrario, descarga data_1.csv y data_2.csv.
    """
    ti = kwargs['ti']
    # Obtener la fecha actual desde el contexto del DAG
    if 'ds' not in kwargs:
        raise ValueError("La fecha actual ('ds') no está disponible en el contexto del DAG.")
    # Convertir la fecha a un objeto datetime
    ds_str = kwargs['ds']
    ds = datetime.strptime(ds_str, '%Y-%m-%d')

    threshold_date = datetime(2024, 11, 1)
    
    if ds < threshold_date:
        ti.xcom_push(key='data_selected', value=f'data_1_{ds_str}')
        return 'download_dataset_1'
    else:
        ti.xcom_push(key='data_selected', value=[f'data_1_{ds_str}', f'data_2_{ds_str}'])
        return ['download_dataset_1', 'download_dataset_2']

def loads_and_merge(**kwargs):
    """
    Carga los archivos data_1.csv y data_2.csv (si está) desde la carpeta 'raw',
    los une en un único DataFrame y lo guarda en 'preprocessed/merged_data.csv'.
    """
    ti = kwargs['ti']

    data_selected = ti.xcom_pull(task_ids='choose_data_branch', key='data_selected')

    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')
    raw_path = os.path.join(main_folder, "raw")
    dataframes = []

    if isinstance(data_selected, str):
        # Si solo se seleccionó un archivo
        file_path = os.path.join(raw_path, f"{data_selected}.csv")
        df = pd.read_csv(file_path)
        dataframes.append(df)

    elif isinstance(data_selected, list):
        # Si se seleccionaron múltiples archivos
        for file_name in data_selected:
            file_path = os.path.join(raw_path, f"{file_name}.csv")
            df = pd.read_csv(file_path)
            dataframes.append(df)
    
    # Concatenar todos los DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)
    preprocessed_path = os.path.join(main_folder, "preprocessed", "merged_data.csv")
    merged_df.to_csv(preprocessed_path, index=False)

def split_data(**kwargs):
    """
    Lee el archivo merged_data.csv desde main_folder/preprocessed,
    divide los datos en train/test (80/20, stratify=HiringDecision, random_state=13),
    y guarda ambos datasets en main_folder/splits.
    """
    ti = kwargs['ti']
    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')

    preprocessed_path = os.path.join(main_folder, "preprocessed", "merged_data.csv")
    df = pd.read_csv(preprocessed_path)
    
    
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

def train_model(**kwargs):
    """
    Entrena un modelo de machine learning (RandomForest, DecisionTree o LogisticRegression)
    y guarda el modelo entrenado en main_folder/models.
    """
    ti = kwargs['ti']
    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')
    model_type = kwargs['op_args'][0]
    model_filename = kwargs['op_args'][1]

    # Leer los datasets de entrenamiento
    splits_path = os.path.join(main_folder, "splits")
    train_df = pd.read_csv(os.path.join(splits_path, "train.csv"))
    X_train = train_df.drop(columns=["HiringDecision"])
    y_train = train_df["HiringDecision"]

    if model_type == 'RandomForest':
        model = RandomForestClassifier(random_state=13)
    elif model_type == 'DecisionTree':
        model = DecisionTreeClassifier(random_state=13)
    elif model_type == 'LogisticRegression':
        model = LogisticRegression(max_iter=1000, random_state=13)
    else:
        raise ValueError(f"Modelo {model_type} no soportado.")
    
    # Crear un pipeline con ColumnTransformer
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer(
            transformers=[
                ("num", "passthrough", X_train.select_dtypes(include=['int64', 'float64']).columns),
                ("cat", "passthrough", X_train.select_dtypes(include=['object']).columns)
            ]
        )),
        ("classifier", model)
    ])
    
    # Entrenar el modelo
    pipeline.fit(X_train, y_train)

    # Guardar el pipeline entrenado
    models_path = os.path.join(main_folder, "models")
    model_path = os.path.join(models_path, model_filename)
    joblib.dump(pipeline, model_path)
    

def evaluate_models(**kwargs):
    """
    Evalúa los modelos entrenados en main_folder/models usando el dataset de test
    y guarda las métricas de evaluación en main_folder/models/evaluation.txt.
    """
    ti = kwargs['ti']
    main_folder = ti.xcom_pull(task_ids='create_folders', key='main_folder')

    # Leer el dataset de test
    splits_path = os.path.join(main_folder, "splits")
    test_df = pd.read_csv(os.path.join(splits_path, "test.csv"))
    X_test = test_df.drop(columns=["HiringDecision"])
    y_test = test_df["HiringDecision"]

    models_path = os.path.join(main_folder, "models")
    evaluation_file = os.path.join(models_path, "evaluation.txt")

    with open(evaluation_file, 'w') as f:
        for model_filename in os.listdir(models_path):
            if model_filename.endswith('.joblib'):
                model_path = os.path.join(models_path, model_filename)
                model = joblib.load(model_path)
                
                # Hacer predicciones
                y_pred = model.predict(X_test)
                
                # Calcular métricas
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='weighted')
                
                # Escribir resultados
                f.write(f"Modelo: {model_filename}\n")
                f.write(f"Accuracy: {accuracy:.4f}\n")
                f.write(f"F1 Score: {f1:.4f}\n\n")