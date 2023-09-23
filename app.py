from flask import Flask,render_template,request,app
from flask import Response
import pickle
import pandas as pd
import numpy as np

application = Flask(__name__)
app = application

## import StandardScaler model and Pridection pickle
Standard_Scaler = pickle.load(open('Models/standardScalar.pkl','rb')) 
Prediction = pickle.load(open('Models/Prediction.pkl','rb'))

## Route for home page
@app.route("/")
def index():
    return render_template('index.html')

#Route the single data point prediction
@app.route('/predict', methods = ['GET','POST'])
def predict_datapoint():
    result = ''

    if request.method == 'POST':
        Pregnancies = float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data_scaled = Standard_Scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict = Prediction.predict(new_data_scaled)

        if predict[0] == 1:
            result = 'Diabetic'
        else:
            result = "Non-Diabetic"

        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
