from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd
import logging

# Set up logging to track which model is being used
logging.basicConfig(level=logging.INFO)

# Load models and scalers
with open("./models/linear_regression_model.pkl", 'rb') as lr_model_file:
    lr_model = pickle.load(lr_model_file)

with open("./models/lr_scaler.pkl", 'rb') as lr_scaler_file:
    lr_scaler = pickle.load(lr_scaler_file)

with open("./models/rf_regression_model.pkl", 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)

with open("./models/rf_scaler.pkl", 'rb') as rf_scaler_file:
    rf_scaler = pickle.load(rf_scaler_file)

# Initialize the Flask app
app = Flask(__name__)

def preprocess_input(data, model, scaler):
    # Convert the input data into a DataFrame for easier manipulation
    df = pd.DataFrame([data])

    # One-hot encoding for categorical features
    categorical_features = ['Gender', 'Occupation', 'Location', 'Coverage Type',
                            'Additional Riders', 'Medical History', 'Lifestyle Factors',
                            'Driving History']
    df = pd.get_dummies(df, columns=categorical_features)

    # Add missing columns that were present during training
    missing_cols = [col for col in model.feature_names_in_ if col not in df.columns]
    for col in missing_cols:
        df[col] = 0

    # Ensure the order of columns matches the training data
    df = df[model.feature_names_in_]

    # Scale numerical features using the scaler
    numerical_features = ['Age', 'Income Level', 'Coverage Amount', 'Deductible', 'Policy Term', 'Claims History']
    df[numerical_features] = scaler.transform(df[numerical_features])

    return df.values

# Route for serving the index.html file from outside the templates folder
@app.route('/')
def index():
    return send_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        model_type = data.get('model')
        
        if model_type == 'linear_regression':
            model = lr_model
            scaler = lr_scaler            
        else:
            model = rf_model
            scaler = rf_scaler

        print("model:", model)
        print("scaler:", scaler)

        # Preprocess the data based on the selected model and scaler
        input_data = preprocess_input(data, model, scaler)

        # Make prediction using the selected model
        prediction = model.predict(input_data)

        return jsonify(prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
