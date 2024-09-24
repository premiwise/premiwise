from flask import Flask, request, jsonify, send_file
import pickle
import numpy as np
import pandas as pd

#linear regression model
# with open("./models/linear_regression_model.pkl", 'rb') as model_file:
#     model = pickle.load(model_file)

# # Load the scaler used during training (if you used scaling for numerical features)
# with open("./models/lr_scaler.pkl", 'rb') as scaler_file:
#     scaler = pickle.load(scaler_file)

#random forest model

# Load your machine learning model from the pickle file
with open("./models/rf_regression_model.pkl", 'rb') as model_file:
    model = pickle.load(model_file)

# Load the scaler used during training (if you used scaling for numerical features)
with open("./models/rf_scaler.pkl", 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Initialize the Flask app
app = Flask(__name__)

# Define a function to preprocess the input data
def preprocess_input(data):
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

# Route to handle form submission and prediction via API
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the POST request
        data = request.get_json(force=True)
        
        # Convert the data to the expected format
        input_data = preprocess_input(data)
        
        # Make prediction using the loaded model
        prediction = model.predict(input_data)
        print(prediction)
        # Return the prediction as JSON
        return jsonify(prediction[0])

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)