from fastapi import FastAPI
import uvicorn
import pickle
import pandas as pd
from pydantic import BaseModel




class parametros(BaseModel): #DEFINICION DE LOS PARAMETROS DEL MODELO, PARA QUE SEAN UTILIZADOS EN EL POST DEL PROGRAMA
  ph: float
  Hardness: float
  Solids: float
  Chloramines: float
  Sulfate: float
  Conductivity: float
  Organic_carbon: float
  Trihalomethanes: float
  Turbidity: float

app = FastAPI() #CREACION DE APP


try:
  with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)
except FileNotFoundError:
  model = None
  print("El archivo models/best_model.pkl no existe.")



@app.get("/")
async def home():
  return {
      "Mensaje": "Bievenid@s, esta es una API para predicción de potabilidad del agua",
      "Descripción": "Esta predicción usa un modelo XGBoost con optimización de hiperparametros desde optuna, este modelo utiliza distintos parametros del agua",
      "Input": {
          "tipo": "objecto JSON, base de datos con los siguientes parametros",
          "parametros":[
              "ph",
              "Hardness",
              "Solids",
              "Chloramines",
              "Sulfate",
              "Conductivity",
              "Organic_carbon",
              "Trihalomethanes",
              "Turbidity"
          ]},
      "Output": {
          "tipo": "Objeto JSON",
          "parametro": "potabilidad",
          "valores": "1 si es potable, 0 si no lo es"
          }}



@app.post("/potabilidad")
async def predecir_potabilidad(data: parametros):
  if model is None:
    return {"error": "El modelo no está disponible."}
  try:
    input_data = pd.DataFrame([data.dict()])
    prediction_potabilidad = model.predict(input_data)
    return {"potabilidad": int(prediction_potabilidad[0])}
  except Exception as e:
    return {"error": str(e)}


if __name__ == '__main__':
    uvicorn.run("main:app", host="0.0.0.0", port=8000) #se utiliza el puerto default 8000.
