from flask import Flask, request, render_template
import pickle
import numpy as np
import streamlit as st
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained model and scaler
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('classifier.pkl', 'rb') as f:
    classifier=pickle.load(f)

# Home route to render the input form
@app.route('/')
def home():
    return render_template('index.html')  # Ensure your HTML file is in the 'templates' folder


# Prediction route to handle form data
@app.route('/submit', methods=['POST'])
def predict():
    try:
        # Retrieve form data
        Pregnancies = int(request.form['pregnancies'])
        Glucose = int(request.form['glucose'])
        BloodPressure = int(request.form['bloodPressure'])
        SkinThickness = int(request.form['skinThickness'])
        Insulin = int(request.form['insulin'])
        BMI = float(request.form['bmi'])
        DiabetesPedigreeFunction = float(request.form['diabetesPedigreeFunction'])
        Age = int(request.form['age'])

        # Prepare data for prediction
        input_data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness,
                                Insulin, BMI, DiabetesPedigreeFunction, Age]])
        standardized_data = scaler.transform(input_data)
        prediction = classifier.predict(standardized_data)
        # probability =classifier.predict_proba(standardized_data)

        # Return result
        return f"""
            <h1>Prediction: {"Diabetic" if prediction[0] == 1 else "Non-Diabetic"}</h1>
            <a href="/">Go Back</a>
        """
    except Exception as e:
        return f"<h1>Error: {str(e)}</h1><a href='/'>Go Back</a>"


# Run the app
if __name__ == "__main__":
    app.run(debug=True)
