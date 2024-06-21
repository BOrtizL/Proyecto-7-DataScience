
from flask import Flask, request, jsonify
from categorizador.predictor import Predictor
import os
import pandas as pd
import pickle

app = Flask(__name__)

# Ruta al archivo pickle de datos y al modelo entrenado
DATA_PATH = '/content/drive/MyDrive/Proyecto 7/data/df_concatenado.pkl'
MODEL_FILENAME = '/content/drive/MyDrive/Proyecto 7/models/best_model.pkl'  # Corregir la ruta del modelo

# Crear la instancia del predictor
predictor = Predictor(data_path=DATA_PATH)

# Inicializar el modelo dentro de la función main
if __name__ == '__main__':
    # Preprocesar datos y cargar modelo
    predictor.preprocess_data()

    # Cargar modelo desde archivo pickle
    with open(MODEL_FILENAME, 'rb') as f:
        loaded_model = pickle.load(f)
        predictor.model = loaded_model  # Asignar el modelo cargado al predictor

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

    # Ejecutar la aplicación Flask
    app.run(host='0.0.0.0', port=5000)
