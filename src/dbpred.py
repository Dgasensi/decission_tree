from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np

# inicializamos Flask
app = Flask(__name__)

# Cargar el modelo entrenado y el imputador k-NN
model = joblib.load('diabetes_model.joblib')
imputer = joblib.load('knn_imputer.joblib')
# definir los nombres de las caracteristicas exactos y en orden
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'] 
# 
@app.route('/', methods=['GET'])
def index():
    # Renderizar el formulario cuando se accede por GET
    return render_template('form.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener los datos del formulario y asegurarse de que los nombres de las características estén alineados
    # Tomamos los datos del formulario y los convertimos en un diccionario
    data = request.form.to_dict()   
    # Convertir los valores de los datos en flotantes o NaN si están vacíos, para el imputador
    data_for_imputation = {k: [float(v) if v != '' else np.nan] for k, v in data.items()}

    # Asegurarse de que los datos para la predicción tengan los nombres de las características
    df = pd.DataFrame.from_dict(data_for_imputation, orient='index').transpose()
    df.columns = feature_names  # Esto alinea los nombres de las columnas con el modelo entrenado

    # Aplicar la imputación k-NN
    imputed_data = imputer.transform(df)

    # Hacer la predicción usando el modelo
    prediction = model.predict(imputed_data)

    # Generar una respuesta
    response = f"{'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}"
    return render_template('form.html', response=response)

if __name__ == '__main__':
    # modo depppuracion, permite que el archivo se actualice automaticamente cuando se haga un cambio y ver mensajes de error
    app.run(debug=True)


