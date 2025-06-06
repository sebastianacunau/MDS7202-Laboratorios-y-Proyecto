import pandas as pd
import mlflow
import pickle
import optuna
import os
import kaleido
import matplotlib.pyplot as plt
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from mlflow.exceptions import MlflowException
import mlflow.sklearn


def get_best_model(experiment_id):
    runs = mlflow.search_runs(experiment_id)
    best_run = runs.sort_values("metrics.valid_f1", ascending=False).iloc[0]
    best_model_id = best_run["run_id"]
    best_model = mlflow.sklearn.load_model("runs:/" + best_model_id + "/model")
    return best_model


def optimize_model():
  df = pd.read_csv('water_potability.csv')
  le = LabelEncoder()
  y = le.fit_transform(df['Potability'])
  X = df.drop('Potability', axis=1)
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


  mlflow.autolog()
  nombre_experimento = f"ESTUDIO DE MODELO DE XGBOOST CON OPTIMIZACION PARA POTABILIDAD DE AGUA "

  try: #CODIGO PARA CHEQUEAR SI EXPERIMENTO YA ESTA CREADO
      experimento = mlflow.get_experiment_by_name(nombre_experimento)
      if experimento is None:
          experimento_id = mlflow.create_experiment(nombre_experimento) #CREACION EXPERIMENTO
      else:
          experimento_id = experimento.experiment_id
  except MlflowException as e:
      experimento = mlflow.get_experiment_by_name(nombre_experimento)
      if experimento is None:
          raise e
      experimento_id = experimento.experiment_id



  def objetivo(trial):
    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.001, 0.1),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
        "gamma": trial.suggest_float("gamma", 0, 1),
        "n_estimators": trial.suggest_int("n_estimators", 10, 300),
        "num_class": len(le.classes_)
    }


#inicio run creacion modelo
    run_name = f" Trial de XGBoost # {trial.number}, con max_depth = {params['max_depth']:.4f}, learning_rate = {params['learning_rate']:.4f}"

    with mlflow.start_run(run_name = run_name, experiment_id = experimento_id, nested=True) as run:

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("valid_f1", f1) #log de metric f1 score
        mlflow.sklearn.log_model(model, "model")


    return f1

  study = optuna.create_study(direction="maximize") #OPTIMIZACION CON OPTUNA
  study.optimize(objetivo, n_trials=25) # optimizacion maximizando con la funcion objetivo

#Punto 1 CHECK
#Punto 2 CHECK
  #Inicio run visualizacion y guardado de plots
  run_name1 = "Visualizacion de Optimizacion con Optuna" #inico 2do run
  with mlflow.start_run(run_name = run_name1, experiment_id = experimento_id):
    historial_optimizacion = plot_optimization_history(study) #CREACION GRAFICOS CON OPTUNA
    plot_coordenadas_paralelas = plot_parallel_coordinate(study)
    plot_importancia_parametros  = plot_param_importances(study)

    if not os.path.exists("plots"): #CREACION CARPETA PLOTS
      os.makedirs("plots")

    if 'kaleido' in globals() and kaleido:
        historial_optimizacion.write_image("plots/historial_optimizacion.png")
        plot_coordenadas_paralelas.write_image("plots/plot_coordenadas_paralelas.png")
        plot_importancia_parametros.write_image("plots/plot_importancia_parametros.png")

    mlflow.log_artifact("plots/historial_optimizacion.png", artifact_path="plots") #LOG DE IMAGENES
    mlflow.log_artifact("plots/plot_coordenadas_paralelas.png", artifact_path= "plots")
    mlflow.log_artifact("plots/plot_importancia_parametros.png", artifact_path= "plots")


#PUNTO 3 CHECK


  best_model = get_best_model(experimento_id)
  run_name2 = "Mejor Modelo" #Inicio 3er run
  with mlflow.start_run(run_name = run_name2, experiment_id = experimento_id):


    if not os.path.exists("models"): #CREACION DE DIRECTORIO MODELS
      os.makedirs("models")

    with open("models/best_model.pkl", "wb") as f:
      pickle.dump(best_model, f)

    mlflow.log_artifact("models/best_model.pkl", artifact_path="models") #LOG DEL MEJOR MODELO EN DIRECTORIO MODELS



#PUNTO 4 CHECK

    best_params = best_model.get_params()
    feature_importances = best_model.feature_importances_
    feature_names = X.columns
    feature_importances = pd.Series(feature_importances, index=feature_names)
    feature_importances.sort_values(inplace=True)
    plt.figure(figsize=(10, 6))
    plt.bar(feature_importances.index, feature_importances.values)
    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Feature Importances")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig("plots/feature_importances.png")
    mlflow.log_artifact("plots/feature_importances.png", artifact_path="plots")

    with open("plots/best_model_params.txt", "w") as f:
      for key, value in best_params.items():
        f.write(f"{key}: {value}\n")
    mlflow.log_artifact("plots/best_model_params.txt", artifact_path="plots")

#PUNTO 7 CHECK

  return best_model


if __name__ == "__main__":
  optimize_model()
