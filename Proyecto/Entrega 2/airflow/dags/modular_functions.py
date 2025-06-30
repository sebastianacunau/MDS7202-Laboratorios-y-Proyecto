
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
    transactions_df['semana'] = transactions_df['purchase_date'].dt.week
    transactions_df['año'] = transactions_df['purchase_date'].dt.year


    
    return {
        'transactions': transactions_df,
        'customers': customers_df,
        'products': products_df
    }
    

def standardize_and_prepare_data(**kwargs):
    ti = kwargs['ti']

    print("Estandarizando y preparando los datos...")

    transactions_df = ti.xcom_pull(task_ids='retrieve_data', key='transactions')
    customers_df = ti.xcom_pull(task_ids='retrieve_data', key='customers')
    products_df = ti.xcom_pull(task_ids='retrieve_data', key='products')

    transactions_df['purchase_date'] = pd.to_datetime(transactions_df['purchase_date'])
    transactions_df['año'] = transactions_df['purchase_date'].dt.year
    transactions_df['semana'] = transactions_df['purchase_date'].dt.isocalendar().week

    transactions_df = transactions_df.drop_duplicates()
    customers_df = customers_df.drop_duplicates()
    products_df = products_df.drop_duplicates()

    if 'num_visit_per_week' in customers_df.columns:
        customers_df = customers_df.drop(columns=['num_visit_per_week', 'region_id', 'zone_id'], errors='ignore')

    df = transactions_df.merge(customers_df, on='customer_id', how='left')
    df = df.merge(products_df, on='product_id', how='left')

    df_agrupado = df.groupby(['customer_id', 'product_id', 'semana', 'año'])['order_id'].count().reset_index()
    df_agrupado.rename(columns={'order_id': 'cantidad_order'}, inplace=True)

    customer_unicos = df['customer_id'].unique()
    productos_unicos = df['product_id'].unique()
    semanas_unicas = df['semana'].unique()
    año_unicos = df['año'].unique()

    customer_unicos_df = pd.DataFrame({'customer_id': customer_unicos})
    productos_unicos_df = pd.DataFrame({'product_id': productos_unicos})
    semanas_unicas_df = pd.DataFrame({'semana': semanas_unicas})
    año_unicos_df = pd.DataFrame({'año': año_unicos})

    combinaciones = customer_unicos_df.merge(productos_unicos_df, how='cross')
    combinaciones = combinaciones.merge(semanas_unicas_df, how='cross')
    combinaciones = combinaciones.merge(año_unicos_df, how='cross')

    data = combinaciones.merge(df_agrupado, on=['customer_id', 'product_id', 'semana', 'año'], how='left')
    data['cantidad_order'] = data['cantidad_order'].fillna(0)
    data['label'] = (data['cantidad_order'] > 0).astype(int)

    data_final = data.merge(customers_df, on='customer_id', how='left')
    data_final = data_final.merge(products_df, on='product_id', how='left')

    print("Datos estandarizados y preparados correctamente.")

    return data_final


#-------------------------------------------------------------------------------------------



def detect_data_drift(**kwargs):
    ti = kwargs['ti']
    new_data = ti.xcom_pull(task_ids='standardize_and_prepare_data')
    reference_data = new_data.sample(n=min(1000, len(new_data)), random_state=42)

    drift_detected = False
    drifted_features = []

    for col in ['cantidad_order']:
        if col in new_data.columns:
            stat, p_value = ks_2samp(reference_data[col], new_data[col])
            if p_value < 0.05:
                drift_detected = True
                drifted_features.append(col)

    print("Drift detectado en:", drifted_features if drift_detected else "Ninguna variable")
    ti.xcom_push(key='drift_detected', value=drift_detected)
    return drift_detected



def choose_drift_branch(**kwargs):
    """
    Elige la rama del DAG según si se detecta drift de datos o no.
    Si se detecta drift, retorna 'retrain_model', de lo contrario, retorna 'skip_training'.
    """
    ti = kwargs['ti']

    print("Eligiendo rama según detección de drift...")

    drift_detected = ti.xcom_pull(task_ids='detect_data_drift', key='drift_detected')

    branch = 'retrain_model' if drift_detected else 'skip_training'
    print(f"Resultado de branching: {branch}")
    return branch


    
def split_data(**kwargs):
    """
    Divide el dataset preprocesado en conjuntos de entrenamiento, validación y prueba.
    Retorna X_train, X_val, X_test, y_train, y_val, y_test a través de XCom.
    """
    from sklearn.model_selection import train_test_split

    ti = kwargs['ti']
    data = ti.xcom_pull(task_ids='standardize_and_prepare_data')

    # 1. División principal: 70% train, 30% val+test
    train_data, val_test_data = train_test_split(
        data, test_size=0.3, random_state=42, shuffle=False
    )

    # 2. División secundaria: 15% val, 15% test
    val_data, test_data = train_test_split(
        val_test_data, test_size=0.5, random_state=42, shuffle=False
    )

    # Features y etiquetas
    drop_cols = ['label', 'product_id', 'customer_id']
    target_col = 'label'

    X_train = train_data.drop(columns=drop_cols)
    y_train = train_data[target_col]

    X_val = val_data.drop(columns=drop_cols)
    y_val = val_data[target_col]

    X_test = test_data.drop(columns=drop_cols)
    y_test = test_data[target_col]

    # Guardar en XCom
    ti.xcom_push(key='X_train', value=X_train)
    ti.xcom_push(key='X_val', value=X_val)
    ti.xcom_push(key='X_test', value=X_test)
    ti.xcom_push(key='y_train', value=y_train)
    ti.xcom_push(key='y_val', value=y_val)
    ti.xcom_push(key='y_test', value=y_test)

    print("Datos divididos correctamente.")


def optimize_model(**kwargs):
    """
    Optimiza hiperparámetros de un RandomForestClassifier usando Optuna.
    Guarda el mejor modelo (pipeline completo) en airflow/models/best_model.pkl
    """
    import optuna
    import joblib
    import os
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, KBinsDiscretizer
    from sklearn.impute import SimpleImputer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score
    from sklearn.model_selection import train_test_split
    from optuna.pruners import MedianPruner

    ti = kwargs["ti"]
    
    # Extraer conjuntos desde XCom
    X_train = ti.xcom_pull(task_ids='split_data', key='X_train')
    y_train = ti.xcom_pull(task_ids='split_data', key='y_train')
    X_val   = ti.xcom_pull(task_ids='split_data', key='X_val')
    y_val   = ti.xcom_pull(task_ids='split_data', key='y_val')

    # Reducimos X_train pero estratificado (solo para optimizar velocidad)
    X_train_sub, _, y_train_sub, _ = train_test_split(
        X_train, y_train, train_size=0.2, stratify=y_train, random_state=42
    )

    def cambios_preprocessor(n_bins, X_subset):
        cat = X_subset.select_dtypes(include=['object']).columns.tolist()
        num = X_subset.select_dtypes(include=['int64', 'int32', 'UInt32']).columns.tolist()
        cont = X_subset.select_dtypes(include=['float64']).columns.tolist()

        num_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler())
        ])
        kbins_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('kbins', KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile'))
        ])
        cat_pipeline = Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_pipeline, num),
            ('cont', kbins_pipeline, cont),
            ('cat', cat_pipeline, cat)
        ])
        return preprocessor

    def objective(trial):
        n_bins       = trial.suggest_int("n_bins", 3, 8)
        n_estimators = trial.suggest_int("n_estimators", 50, 100, step=10)
        max_depth    = trial.suggest_int("max_depth", 5, 10)
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])
        criterion    = trial.suggest_categorical("criterion", ["gini", "entropy"])

        preprocessor = cambios_preprocessor(n_bins, X_train_sub)

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            max_features=max_features,
            criterion=criterion,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        )

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('modelo', model)
        ])

        pipeline.fit(X_train_sub, y_train_sub)
        y_pred = pipeline.predict(X_val)
        f1 = f1_score(y_val, y_pred, pos_label=1, average='weighted')

        trial.report(f1, step=0)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return f1

    study = optuna.create_study(
        direction="maximize",
        pruner=MedianPruner(n_startup_trials=3)
    )
    study.optimize(objective, n_trials=20)

    # Reentrenar con el mejor modelo usando todo X_train
    best_params = study.best_params
    print("Mejores parámetros encontrados:", best_params)

    final_preprocessor = cambios_preprocessor(best_params['n_bins'], X_train)
    final_model = RandomForestClassifier(
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        max_features=best_params['max_features'],
        criterion=best_params['criterion'],
        class_weight="balanced",
        n_jobs=-1,
        random_state=42
    )

    best_pipeline = Pipeline([
        ('preprocessor', final_preprocessor),
        ('modelo', final_model)
    ])
    best_pipeline.fit(X_train, y_train)

    # Guardar el modelo completo
    path = os.path.join("airflow", "models", "best_model.pkl")
    joblib.dump(best_pipeline, path)
    print(f"Modelo optimizado guardado en: {path}")

    return best_pipeline


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
    ti = kwargs['ti']
    model = ti.xcom_pull(task_ids='optimize_model')
    X_val = ti.xcom_pull(task_ids='split_data', key='X_val')

    X_val_transformed = model.named_steps['preprocessor'].transform(X_val)
    explainer = shap.Explainer(model.named_steps['modelo'], X_val_transformed)
    shap_values = explainer(X_val_transformed)

    os.makedirs("airflow/interpretation", exist_ok=True)
    path = "airflow/interpretation/shap_summary_plot.png"
    shap.summary_plot(shap_values, show=False)
    plt.savefig(path)
    print(f"Gráfico SHAP guardado en: {path}")

    mlflow.set_experiment("SodAI_Model_Experiments")
    with mlflow.start_run(run_name="SHAP_Interpretation"):
        mlflow.log_artifact(path)
    
    return interpretations
def evaluate_model(**kwargs):
    ti = kwargs['ti']
    pipeline = ti.xcom_pull(task_ids='optimize_model')
    X_test = ti.xcom_pull(task_ids='split_data', key='X_test')
    y_test = ti.xcom_pull(task_ids='split_data', key='y_test')

    

    y_pred = pipeline.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted'),
        "recall": recall_score(y_test, y_pred, average='weighted'),
        "f1_score": f1_score(y_test, y_pred, average='weighted'),
        
    }

    mlflow.set_experiment("SodAI_Model_Experiments")
    with mlflow.start_run(run_name="Evaluate_Model"):
        mlflow.log_metrics(results)

    print("Evaluación del modelo:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


def generate_predictions(**kwargs):
    """
    Genera un archivo CSV con los pares cliente-producto que tienen predicción positiva (label=1)
    utilizando el modelo entregado por optimize_model (vía XCom).
    """
    import pandas as pd
    import os

    ti = kwargs['ti']

    # Recuperar pipeline entrenado desde optimize_model
    model = ti.xcom_pull(task_ids='optimize_model')
    if model is None:
        raise ValueError("Modelo no disponible desde XCom")

    # Recuperar datos originales preprocesados
    full_data = ti.xcom_pull(task_ids='standardize_and_prepare_data')

    # Separar identificadores
    id_cols = ['customer_id', 'product_id']
    if not set(id_cols).issubset(full_data.columns):
        raise ValueError("No se encontraron columnas customer_id y product_id en los datos")

    features = full_data.drop(columns=['label'] + id_cols, errors='ignore')
    ids = full_data[id_cols]

    # Generar predicciones
    preds = model.predict(features)

    # Filtrar solo predicciones positivas (label=1)
    positive_preds = ids[preds == 1]

    # Guardar CSV
    output_path = os.path.join("airflow", "predictions", "positive_predictions.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    positive_preds.to_csv(output_path, index=False)

    print(f"Predicciones positivas guardadas en: {output_path}")
