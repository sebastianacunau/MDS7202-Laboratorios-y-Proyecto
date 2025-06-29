import pandas as pd
import os
import itertools

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from XGBoost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_curve, log_loss


def retrieve_data(**kwargs):
    """
    Recupera los datos de transacciones, clientes y productos desde el directorio de trabajo y los carga en DataFrames.
    Se realiza un preprocesamiento básico para dejar el dataset agregado por semana y año,
    y eliminar columnas con un único valor en los DataFrames de clientes y productos.
    Retorna un DataFrame con todas las combinaciones cliente-producto-semana para realizar análisis.
    """
    # Se recuperan los datos de transacciones, clientes y productos desde el directorio de trabajo.
    exec_date = kwargs.get('ds')
    
    print(f"Extrayendo datos para la fecha: {exec_date}")

    print("Extrayendo 'transacciones.parquet' ...")
    transactions_path = os.path.join("data", "transacciones.parquet")
    transactions_df = pd.read_parquet(transactions_path)

    print("Extrayendo 'clientes.parquet' ...")
    customers_path = os.path.join("data", "clientes.parquet")
    customers_df = pd.read_parquet(customers_path)

    print("Extrayendo 'productos.parquet' ...")
    products_path = os.path.join("data", "productos.parquet")
    products_df = pd.read_parquet(products_path)
    
    print("Datos extraídos correctamente.")

    # Se eliminan columnas con un único valor en los DataFrames de clientes y productos
    customers_df = customers_df.loc[:, customers_df.nunique() > 1]
    products_df = products_df.loc[:, products_df.nunique() > 1]
    
    # Se agregan columnas de semana del año y año a transactions_df
    transactions_df['purchase_date'] = pd.to_datetime(transactions_df['purchase_date'])
    transactions_df['week_of_year'] = transactions_df['purchase_date'].dt.week
    transactions_df['year'] = transactions_df['purchase_date'].dt.year

    # Se realiza una agregación de transacciones por cliente, producto, semana y año
    transactions_df = transactions_df.groupby(
        ['customer_id', 'product_id', 'week_of_year', 'year']).agg(
            quantity_ordered=('items', 'sum'),
            orders_count=('order_id', 'nunique')
        ).reset_index()
    
    return {
        'transactions': transactions_df,
        'customers': customers_df,
        'products': products_df
    }
    

def standardize_and_prepare_data(**kwargs):
    """
    Estandariza y prepara los datos extraídos, de acuerdo al preprocesamiento detallado en la entrega 1.
    """
    ti = kwargs['ti']
    
    print("Estandarizando y preparando los datos...")
    
    transactions_df = ti.xcom_pull(task_ids='retrieve_data', key='transactions')
    customers_df = ti.xcom_pull(task_ids='retrieve_data', key='customers')
    products_df = ti.xcom_pull(task_ids='retrieve_data', key='products')

    # Se realiza un merge de los DataFrames de transacciones, clientes y productos, de modo que estén todas las 
    # combinaciones cliente-producto-semana disponibles.
    customer_ids = customers_df['client_id'].unique()
    product_ids = products_df['product_id'].unique()
    years = transactions_df['year'].unique()
    weeks = range(1, 53)  # Assuming up to 52 weeks in a year, adjust if needed
    
    # Se crea el producto cartesiano de todas las combinaciones cliente-producto-año-semana
    all_combinations = pd.DataFrame(
        list(itertools.product(customer_ids, product_ids, years, weeks)),
        columns=['client_id', 'product_id', 'year', 'week_of_year']
    )
    
    # Se realiza un merge con las transacciones agregadas para obtener los datos completos
    complete_data = pd.merge(
        all_combinations, 
        transactions_df, 
        how='left', 
        on=['client_id', 'product_id', 'year', 'week_of_year']
    )
    
    # Para las columnas de cantidad de productos pedidos y número de pedidos, se rellenan los valores NaN con 0
    complete_data['quantity_ordered'].fillna(0, inplace=True)
    complete_data['orders_count'].fillna(0, inplace=True)

    print("Datos estandarizados y preparados correctamente.")
    
    return complete_data

def detect_data_drift(**kwargs):
    """
    Detecta el drift de datos en el DataFrame preprocesado.
    Utiliza las técnicas de detección de drift de Kolmogorov-Smirnov y la divergencia KL,
    y retorna un booleano indicando si hay data drift.
    """
    ti = kwargs['ti']
    
    print("Detectando drift de datos...")
    
    new_data = ti.xcom_pull(task_ids='standardize_and_prepare_data')

    # Aquí hay que implementar la lógica para detectar el drift de datos.

    drift_results = {
        "kolmogorov_smirnov": False,  # Reemplazar con el resultado real
        "kl_divergence": False,  # Reemplazar con el resultado real
    }
    
    print("Detección de drift completada.")
    
    return drift_results

def choose_drift_branch(**kwargs):
    """
    Elige la rama del DAG según si se detecta drift de datos o no.
    Si se detecta drift, retorna 'retrain_model', de lo contrario, retorna 'skip_training'.
    """
    ti = kwargs['ti']
    
    print("Eligiendo rama según detección de drift...")

    drift_detected = ti.xcom_pull(task_ids='detect_data_drift', key='drift_detected')
    
    if drift_detected:
        print("Drift detectado, se reentrenará el modelo.")
        return 'retrain_model'
    else:
        print("No se detectó drift, se saltará el entrenamiento del modelo.")
        return 'skip_training'
    
def split_data(**kwargs):

def optimize_model(**kwargs):
    """
    Optimiza el modelo utilizando GridSearchCV o RandomizedSearchCV.
    Retorna el mejor modelo encontrado.
    """
    ti = kwargs['ti']
    
    print("Optimizando el modelo...")

    model = ti.xcom_pull(task_ids='train_model', key='model')
    
    # Aquí hay que implementar la lógica de optimización del modelo.
    
    best_model = model  # Reemplazar con el mejor modelo encontrado
    
    print("Optimización del modelo completada.")
    
    return best_model

def train_model(**kwargs):
    """
    Entrena un modelo de machine learning (RandomForest, DecisionTree o LogisticRegression)
    utilizando los datos preprocesados y retorna el modelo entrenado.
    """
    ti = kwargs['ti']
    
    print("Entrenando el modelo...")

    preprocessed_data = ti.xcom_pull(task_ids='standardize_and_prepare_data')
    
    # Aquí hay que implementar la lógica para entrenar el modelo.
    
    model = ...  # Reemplazar con el modelo entrenado
    
    print("Modelo entrenado correctamente.")
    
    return model

def interpret_model(**kwargs):
    """
    Interpreta el modelo entrenado utilizando SHAP o LIME.
    Retorna las interpretaciones del modelo.
    """
    ti = kwargs['ti']
    
    print("Interpretando el modelo...")

    model = ti.xcom_pull(task_ids='train_model')
    
    # Aquí hay que implementar la lógica para interpretar el modelo.
    
    interpretations = ...  # Reemplazar con las interpretaciones del modelo
    
    print("Interpretación del modelo completada.")
    
    return interpretations

def evaluate_model(**kwargs):
    """
    Evalúa el modelo entrenado utilizando métricas como accuracy, precision, recall y F1-score.
    Retorna un diccionario con las métricas de evaluación.
    """
    ti = kwargs['ti']
    
    print("Evaluando el modelo...")

    model = ti.xcom_pull(task_ids='train_model')
    test_data = ti.xcom_pull(task_ids='split_data', key='test_data')
    
    # Aquí hay que implementar la lógica para evaluar el modelo.
    
    evaluation_metrics = {
        "accuracy": 0.95,  # Reemplazar con el valor real
        "precision": 0.90,  # Reemplazar con el valor real
        "recall": 0.85,  # Reemplazar con el valor real
        "f1_score": 0.87,  # Reemplazar con el valor real
    }
    
    print("Evaluación del modelo completada.")
    
    return evaluation_metrics

