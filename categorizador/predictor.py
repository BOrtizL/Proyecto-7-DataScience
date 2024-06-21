
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump, load
import os

class Predictor:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.df_categorica = None
        self.df_copy = None
        self.df_encoded = None
        self.df_concatenado = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.model = None
        self.scaler = None
        self.pca = None
        self.load_data()  # Cargar datos al inicializar

    def load_data(self):
        try:
            with open(self.data_path, 'rb') as f:
                self.df = pickle.load(f)
        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta especificada: {self.data_path}")
        except Exception as e:
            print(f"Error al cargar el archivo pickle: {str(e)}")

    def preprocess_data(self):
        if self.df is None:
            print("Error: No se ha cargado ningún DataFrame.")
            return

        # Seleccionar columnas categóricas
        self.df_categorica = self.df.select_dtypes(include=["object", "bool"]).columns

        # Eliminar columnas específicas si existen
        columns_to_drop = ['biourl', 'movie', "birthplace", "date_of_birth", 'person', "_last_judgment_at", "_unit_id", "birthplace:confidence", "date_of_birth:confidence", "race_ethnicity:confidence", "religion:confidence", "sexual_orientation:confidence", "year_of_award:confidence"]
        self.df_categorica = self.df_categorica.drop(columns_to_drop, errors='ignore')

        # Hacer una copia del DataFrame original
        self.df_copy = self.df.copy()

        # Seleccionar columnas que no sean 'object' ni 'bool'
        self.df_copy = self.df_copy.select_dtypes(exclude=['object', 'bool'])

        # Obtener una copia del DataFrame codificado para las columnas categóricas
        self.df_encoded = self.df[self.df_categorica].copy()

        # Crear embeddings para cada columna categórica
        category_embedding = {}
        for column in self.df_encoded.select_dtypes(include=['object', 'bool']).columns:
            unique_categories = self.df_encoded[column].unique()
            category_embedding[column] = dict(zip(unique_categories, range(1, len(unique_categories) + 1)))
            self.df_encoded[column] = self.df_encoded[column].map(category_embedding[column]).fillna(0)

        # Para finalizar con la transformación de los datos, se concatenan los df: df_encoded con df_copy
        self.df_concatenado = pd.concat([self.df_encoded, self.df_copy], axis=1)

        # Separar características (X) y etiquetas (y)
        self.X = self.df_concatenado.drop('gender', axis=1)
        self.y = self.df_concatenado['gender']

        # Dividir en entrenamiento y prueba
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Escalar características
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        # Aplicar PCA
        self.pca = PCA(n_components=8)  # Número óptimo de componentes
        self.X_train = self.pca.fit_transform(self.X_train)
        self.X_test = self.pca.transform(self.X_test)

    def build_model(self):
        if self.X_train is None or self.y_train is None:
            print("Error: No se han dividido los datos en entrenamiento y prueba.")
            return

        # Crear pipeline con StandardScaler, PCA y LogisticRegression
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=8)),
            ('logreg', LogisticRegression())
        ])

        # Definir el grid de hiperparámetros para LogisticRegression
        param_grid = {
            'logreg__C': [0.1, 1, 10, 100],
            'logreg__solver': ['liblinear', 'lbfgs'],
            'logreg__penalty': ['l2'],
            'logreg__max_iter': [100, 200, 300]
        }

        # Configurar GridSearchCV con validación cruzada estratificada
        grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')

        # Entrenar el modelo
        grid_search.fit(self.X_train, self.y_train)

        # Evaluar el mejor modelo
        self.model = grid_search.best_estimator_
        y_pred = self.model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f'Accuracy: {accuracy}')
        print(classification_report(self.y_test, y_pred))

    def save_model(self, filename):
        try:
            if self.model is None:
                print("Error: No hay ningún modelo entrenado para guardar.")
                return

            if not os.path.exists('saved_models'):
                os.makedirs('saved_models')

            model_path = os.path.join('saved_models', filename + '.joblib')
            dump(self.model, model_path)

            if self.scaler:
                scaler_path = os.path.join('saved_models', filename + '_scaler.joblib')
                dump(self.scaler, scaler_path)

            if self.pca:
                pca_path = os.path.join('saved_models', filename + '_pca.joblib')
                dump(self.pca, pca_path)

            print(f"Modelo guardado correctamente en '{model_path}'")

        except Exception as e:
            print(f"Error al guardar el modelo: {str(e)}")

    def load_model(self, filename):
        try:
            model_path = os.path.join('saved_models', filename + '.joblib')
            self.model = load(model_path)

            scaler_path = os.path.join('saved_models', filename + '_scaler.joblib')
            if os.path.exists(scaler_path):
                self.scaler = load(scaler_path)

            pca_path = os.path.join('saved_models', filename + '_pca.joblib')
            if os.path.exists(pca_path):
                self.pca = load(pca_path)

            print(f"Modelo '{filename}' cargado correctamente.")

        except FileNotFoundError:
            print(f"Error: No se encontró el archivo en la ruta especificada: {filename}")
        except Exception as e:
            print(f"Error al cargar el modelo: {str(e)}")

# Ejemplo de uso:
data_path = r'/content/drive/My Drive/Proyecto 7/data/df.pkl'  # Ruta hacia tu archivo pickle
predictor = Predictor(data_path=data_path)
predictor.preprocess_data()
predictor.build_model()

# Guardar el modelo
predictor.save_model('logistic_regression_model')

# Cargar el modelo
loaded_predictor = Predictor(data_path=data_path)
loaded_predictor.load_model('logistic_regression_model')
# Ahora loaded_predictor.model está listo para hacer predicciones o evaluaciones.
