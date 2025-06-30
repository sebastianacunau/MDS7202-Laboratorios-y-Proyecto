import gradio as gr
import requests

API_URL = "http://backend:8000/predict"  # El nombre del servicio backend en docker-compose

def predict(year, week, cantidad_order, customer_type, Y, X, num_deliver, brand, category, sub_category, segment, package, size):
    input_data = {
        "year": int(year),
        "week": int(week),
        "cantidad_order": float(cantidad_order),
        "customer_type": customer_type,
        "Y": float(Y),
        "X": float(X),
        "num_deliver_per_week": int(num_deliver),
        "brand": brand,
        "category": category,
        "sub_category": sub_category,
        "segment": segment,
        "package": package,
        "size": float(size),
    }
    try:
        response = requests.post(API_URL, json=input_data)
        prediction = response.json().get("prediction")
        return f"¿Comprará?: {'Sí' if prediction == 1 else 'No'}"
    except Exception as e:
        return f"Error: {e}"

demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Number(label="Año"),
        gr.Number(label="Semana del año"),
        gr.Number(label="Cantidad ordenada en la semana"),
        gr.Textbox(label="Tipo de cliente"),
        gr.Number(label="Coordenada Y"),
        gr.Number(label="Coordenada X"),
        gr.Number(label="Número de despachos por semana"),
        gr.Textbox(label="Marca"),
        gr.Textbox(label="Categoría"),
        gr.Textbox(label="Subcategoría"),
        gr.Textbox(label="Segmento"),
        gr.Textbox(label="Tipo de paquete"),
        gr.Number(label="Tamaño"),
    ],
    outputs=gr.Textbox(label="Predicción"),
    title="Predicción de Compra Semanal",
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
