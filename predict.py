"""
Prediction module for Customer Churn
Loads the trained model and makes predictions
"""

import pickle
import numpy as np

# Load model artifacts
with open('model/churn_model.pkl', 'rb') as f:
    artifacts = pickle.load(f)

model = artifacts['model']
label_encoders = artifacts['label_encoders']

def predict_churn(tenure, monthly_charges, total_charges, contract, internet_service, payment_method):
    """
    Predict customer churn based on input features.
    
    Parameters:
    - tenure: int (months with company)
    - monthly_charges: float
    - total_charges: float
    - contract: str ('Month-to-month', 'One year', 'Two year')
    - internet_service: str ('DSL', 'Fiber optic', 'No')
    - payment_method: str ('Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)')
    
    Returns:
    - prediction: int (0 = No Churn, 1 = Churn)
    - probability: float (probability of churn)
    """
    
    # Encode categorical features
    contract_encoded = label_encoders['Contract'].transform([contract])[0]
    internet_encoded = label_encoders['InternetService'].transform([internet_service])[0]
    payment_encoded = label_encoders['PaymentMethod'].transform([payment_method])[0]
    
    # Create feature array
    features = np.array([[
        tenure,
        monthly_charges,
        total_charges,
        contract_encoded,
        internet_encoded,
        payment_encoded
    ]])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]  # Probability of churn
    
    return int(prediction), float(probability)