
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="SodAI Backend API")


MODEL_PATH = os.path.join("..", "..", "airflow", "models", "best_model.pkl")


try:
    model_pipeline = joblib.load(MODEL_PATH)
    print("Modelo cargado exitosamente.")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

class PredictionRequest(BaseModel):
    semana: int
    año: int
    cantidad_order: float
    customer_type: str
    Y: float
    X: float
    num_deliver_per_week: int
    brand: str
    category: str
    sub_category: str
    segment: str
    package: str
    size: float

@app.get('/') # ruta
async def home(): 
    return {'Bienvenidos a la API de SodAI Drinks en su version 1.0, agrega la ruta "/predict" para hacer predicciones'}



@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        input_df = pd.DataFrame([request.dict()])
        prediction = model_pipeline.predict(input_df)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
