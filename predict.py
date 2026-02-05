"""
Prediction module for E-Commerce Customer Churn
Loads the trained model and makes predictions with risk analysis
"""

import pickle
import numpy as np
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model artifacts
try:
    with open('model/churn_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    
    model = artifacts['model']
    label_encoders = artifacts['label_encoders']
    feature_names = artifacts['feature_names']
    categorical_cols = artifacts['categorical_cols']
    numerical_cols = artifacts['numerical_cols']
    
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    raise

# Valid values for categorical features
VALID_VALUES = {
    'preferred_login_device': ['Mobile Phone', 'Computer', 'Phone'],
    'preferred_payment_mode': ['Credit Card', 'Debit Card', 'UPI', 'Cash on Delivery', 'E wallet', 'CC', 'COD'],
    'gender': ['Male', 'Female'],
    'prefered_order_cat': ['Laptop & Accessory', 'Mobile', 'Mobile Phone', 'Fashion', 'Grocery', 'Others'],
    'marital_status': ['Single', 'Married', 'Divorced']
}

def validate_input(data):
    """Validate input values before prediction"""
    errors = []
    
    # Validate numerical ranges
    if 'tenure' in data and not (0 <= data['tenure'] <= 100):
        errors.append(f"tenure must be between 0 and 100")
    if 'satisfaction_score' in data and not (1 <= data['satisfaction_score'] <= 5):
        errors.append(f"satisfaction_score must be between 1 and 5")
    if 'complain' in data and data['complain'] not in [0, 1]:
        errors.append(f"complain must be 0 or 1")
    
    return errors

def predict_churn(tenure, warehouse_to_home, hour_spend_on_app, number_of_device_registered,
                  satisfaction_score, number_of_address, complain, order_amount_hike,
                  coupon_used, order_count, day_since_last_order, cashback_amount,
                  city_tier, preferred_login_device, preferred_payment_mode, gender,
                  prefered_order_cat, marital_status):
    """
    Predict customer churn based on input features.
    
    Returns:
    - prediction: int (0 = No Churn, 1 = Churn)
    - probability: float (probability of churn)
    - risk_level: str ('Low', 'Medium', 'High', 'Critical')
    - risk_factors: list of identified risk factors
    """
    
    # Encode categorical features
    encoded_values = {}
    
    categorical_mapping = {
        'PreferredLoginDevice': preferred_login_device,
        'PreferredPaymentMode': preferred_payment_mode,
        'Gender': gender,
        'PreferedOrderCat': prefered_order_cat,
        'MaritalStatus': marital_status
    }
    
    for col, value in categorical_mapping.items():
        if col in label_encoders:
            try:
                encoded_values[col] = label_encoders[col].transform([str(value)])[0]
            except ValueError:
                # Handle unseen labels
                encoded_values[col] = 0
                logger.warning(f"Unknown value '{value}' for {col}, using default")
    
    # Build feature array in correct order
    feature_dict = {
        'Tenure': tenure,
        'WarehouseToHome': warehouse_to_home,
        'HourSpendOnApp': hour_spend_on_app,
        'NumberOfDeviceRegistered': number_of_device_registered,
        'SatisfactionScore': satisfaction_score,
        'NumberOfAddress': number_of_address,
        'Complain': complain,
        'OrderAmountHikeFromlastYear': order_amount_hike,
        'CouponUsed': coupon_used,
        'OrderCount': order_count,
        'DaySinceLastOrder': day_since_last_order,
        'CashbackAmount': cashback_amount,
        'CityTier': city_tier,
        **encoded_values
    }
    
    # Create feature array matching training order
    features = []
    for fname in feature_names:
        if fname in feature_dict:
            features.append(feature_dict[fname])
        else:
            features.append(0)
            logger.warning(f"Missing feature {fname}, using 0")
    
    features = np.array([features])
    
    # Make prediction
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    # Determine risk level
    if probability < 0.25:
        risk_level = 'Low'
    elif probability < 0.50:
        risk_level = 'Medium'
    elif probability < 0.75:
        risk_level = 'High'
    else:
        risk_level = 'Critical'
    
    # Identify risk factors
    risk_factors = []
    
    if tenure < 6:
        risk_factors.append(f"New customer (tenure: {tenure} months) - highest churn risk")
    if complain == 1:
        risk_factors.append("Has filed complaints - strong churn indicator")
    if satisfaction_score <= 2:
        risk_factors.append(f"Low satisfaction score ({satisfaction_score}/5)")
    if day_since_last_order > 30:
        risk_factors.append(f"Inactive for {day_since_last_order} days")
    if warehouse_to_home > 30:
        risk_factors.append(f"Long delivery distance ({warehouse_to_home} km)")
    if cashback_amount < 100:
        risk_factors.append("Low cashback engagement")
    if order_count < 2:
        risk_factors.append("Very few orders placed")
    if number_of_address > 5:
        risk_factors.append("Multiple addresses - possible account issues")
    
    if not risk_factors:
        risk_factors.append("No significant risk factors identified")
    
    logger.info(f"Prediction: {prediction}, Probability: {probability:.4f}, Risk: {risk_level}")
    
    return int(prediction), float(probability), risk_level, risk_factors

def get_model_info():
    """Return model metadata"""
    return {
        'model_type': type(model).__name__,
        'features': feature_names,
        'valid_values': VALID_VALUES,
        'categorical_features': categorical_cols,
        'numerical_features': numerical_cols
    }