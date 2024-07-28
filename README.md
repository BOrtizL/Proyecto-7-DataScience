# Proyecto: Análisis Demográfico de los Ganadores del Oscar
Oscar-demographics Dataset Project

## Descripción
Este proyecto realiza un análisis demográfico de los ganadores del Oscar utilizando técnicas de data science. 
Se comparan modelos de clasificación como la Regresión Logística y Random Forest.

El modelo y el procesamiento de datos se implementan en un entorno Flask para proporcionar una interfaz web interactiva.

## Requisitos
- Python 3.x
- pip
- Git

Clona este repositorio en tu máquina local: git clone https://github.com/BOrtizL/Proyecto-7.git

## Configuración del Entorno
1. Crear un Entorno Virtual
Para crear un entorno virtual, abre tu terminal y ejecuta el siguiente comando:
python3 -m venv nombre_del_entorno

2. Activar el Entorno Virtual
En Unix/Linux:source nombre_del_entorno/bin/activate
En Windows:.\nombre_del_entorno\Scripts\activate

3. Instalar Dependencias
Una vez activado el entorno virtual, instala las dependencias necesarias ejecutando:
pip install -r requirements.txt

4. Descargar Datos de Entrenamiento
Descarga el archivo df_concatenado.pkl en el directorio data.

5. Cargar el Modelo y Hacer Predicciones
import pickle

# Ruta al modelo guardado
model_path = 'model/best_model.pkl'

# Cargar el modelo
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Hacer predicciones
datos_nuevos = ...  # Aquí van los datos nuevos a predecir
predictions = model.predict(datos_nuevos)

6. Ejecutar la API Flask
Para ejecutar la API Flask que permite hacer predicciones a través de solicitudes HTTP, asegúrate de que las rutas sean correctas en app.py

El contenido del archivo "categorizador/app.py"

from flask import Flask, request, jsonify
from categorizador.predictor import Predictor
import os
import pandas as pd

app = Flask(__name__)

# Ruta al archivo pickle de datos y al modelo entrenado
DATA_PATH = 'data/df_concatenado.pkl'
MODEL_FILENAME = 'model/best_model.pkl'

# Crear la instancia del predictor
predictor = Predictor(data_path=DATA_PATH)
predictor.preprocess_data()
predictor.load_model(MODEL_FILENAME)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        df = pd.DataFrame(data)
        df_encoded = predictor.scaler.transform(df)
        df_pca = predictor.pca.transform(df_encoded)
        prediction = predictor.model.predict(df_pca)
        response = {'prediction': prediction.tolist()}
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

Para ejecutar la API Flask, usa el siguiente comando: python categorizador/app.py

La API estará disponible en http://0.0.0.0:5000/predict.

Este README proporciona instrucciones claras sobre cómo configurar y usar el entorno virtual, entrenar y guardar modelos, y ejecutar una API Flask para predicciones.








