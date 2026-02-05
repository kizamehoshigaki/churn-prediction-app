"""
Flask API for E-Commerce Customer Churn Prediction
- RESTful endpoints for predictions
- Input validation & error handling
- Structured logging
- Business recommendations
"""

from flask import Flask, request, jsonify
from predict import predict_churn, get_model_info, VALID_VALUES
import os
import logging
from datetime import datetime
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Request counter for metrics
request_count = {'predict': 0, 'health': 0}

LABEL_MAP = {0: "No Churn", 1: "Churn"}

# Business recommendations based on risk factors
RECOMMENDATIONS = {
    'tenure': "Implement onboarding program with early engagement incentives",
    'complain': "Prioritize complaint resolution and follow-up within 24 hours",
    'satisfaction': "Offer personalized discount or loyalty reward",
    'inactive': "Send re-engagement email with exclusive offer",
    'distance': "Offer free shipping or local pickup options",
    'cashback': "Increase cashback percentage for next purchase",
    'orders': "Provide first-time buyer discount on second order"
}

@app.route('/')
def home():
    """API documentation endpoint"""
    return jsonify({
        'name': 'E-Commerce Customer Churn Prediction API',
        'version': '2.0.0',
        'description': 'ML-powered API to predict e-commerce customer churn',
        'model': 'XGBoost with SMOTE balancing',
        'endpoints': {
            '/': 'GET - API documentation',
            '/health': 'GET - Health check',
            '/predict': 'POST - Predict customer churn',
            '/model/info': 'GET - Model information',
            '/model/valid-values': 'GET - Valid input values'
        },
        'author': 'Aaditya Krishna',
        'github': 'https://github.com/kizamehoshigaki/ecommerce-churn-prediction'
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    request_count['health'] += 1
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.utcnow().isoformat(),
        'requests_served': request_count
    })

@app.route('/model/info', methods=['GET'])
def model_info():
    """Return model metadata"""
    try:
        info = get_model_info()
        
        try:
            with open('model/model_metadata.json', 'r') as f:
                metadata = json.load(f)
            info.update(metadata)
        except FileNotFoundError:
            pass
        
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/valid-values', methods=['GET'])
def valid_values():
    """Return valid values for categorical inputs"""
    return jsonify(VALID_VALUES)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict customer churn for e-commerce
    
    Required fields:
    - tenure, warehouse_to_home, hour_spend_on_app, number_of_device_registered
    - satisfaction_score, number_of_address, complain, order_amount_hike
    - coupon_used, order_count, day_since_last_order, cashback_amount
    - city_tier, preferred_login_device, preferred_payment_mode, gender
    - prefered_order_cat, marital_status
    """
    request_count['predict'] += 1
    start_time = datetime.utcnow()
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Required fields
        required_fields = [
            'tenure', 'warehouse_to_home', 'hour_spend_on_app', 
            'number_of_device_registered', 'satisfaction_score', 'number_of_address',
            'complain', 'order_amount_hike', 'coupon_used', 'order_count',
            'day_since_last_order', 'cashback_amount', 'city_tier',
            'preferred_login_device', 'preferred_payment_mode', 'gender',
            'prefered_order_cat', 'marital_status'
        ]
        
        missing_fields = [f for f in required_fields if f not in data]
        if missing_fields:
            return jsonify({
                'error': f'Missing required fields: {missing_fields}',
                'required_fields': required_fields
            }), 400
        
        # Extract features
        prediction, probability, risk_level, risk_factors = predict_churn(
            tenure=int(data['tenure']),
            warehouse_to_home=float(data['warehouse_to_home']),
            hour_spend_on_app=float(data['hour_spend_on_app']),
            number_of_device_registered=int(data['number_of_device_registered']),
            satisfaction_score=int(data['satisfaction_score']),
            number_of_address=int(data['number_of_address']),
            complain=int(data['complain']),
            order_amount_hike=float(data['order_amount_hike']),
            coupon_used=int(data['coupon_used']),
            order_count=int(data['order_count']),
            day_since_last_order=int(data['day_since_last_order']),
            cashback_amount=float(data['cashback_amount']),
            city_tier=int(data['city_tier']),
            preferred_login_device=data['preferred_login_device'],
            preferred_payment_mode=data['preferred_payment_mode'],
            gender=data['gender'],
            prefered_order_cat=data['prefered_order_cat'],
            marital_status=data['marital_status']
        )
        
        # Generate recommendations based on risk factors
        recommendations = []
        for factor in risk_factors:
            if 'tenure' in factor.lower() or 'new customer' in factor.lower():
                recommendations.append(RECOMMENDATIONS['tenure'])
            if 'complaint' in factor.lower():
                recommendations.append(RECOMMENDATIONS['complain'])
            if 'satisfaction' in factor.lower():
                recommendations.append(RECOMMENDATIONS['satisfaction'])
            if 'inactive' in factor.lower() or 'days' in factor.lower():
                recommendations.append(RECOMMENDATIONS['inactive'])
            if 'distance' in factor.lower() or 'delivery' in factor.lower():
                recommendations.append(RECOMMENDATIONS['distance'])
            if 'cashback' in factor.lower():
                recommendations.append(RECOMMENDATIONS['cashback'])
            if 'orders' in factor.lower():
                recommendations.append(RECOMMENDATIONS['orders'])
        
        recommendations = list(set(recommendations))  # Remove duplicates
        
        response_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        response = {
            'prediction': LABEL_MAP.get(prediction, str(prediction)),
            'churn_probability': round(probability, 4),
            'retention_probability': round(1 - probability, 4),
            'risk_level': risk_level,
            'risk_factors': risk_factors,
            'recommendations': recommendations if recommendations else ["Continue current engagement strategy"],
            'customer_summary': {
                'tenure_months': data['tenure'],
                'satisfaction_score': data['satisfaction_score'],
                'has_complained': bool(data['complain']),
                'total_orders': data['order_count'],
                'days_since_last_order': data['day_since_last_order']
            },
            'metadata': {
                'response_time_ms': round(response_time, 2),
                'model_version': '2.0.0',
                'timestamp': datetime.utcnow().isoformat()
            }
        }
        
        logger.info(f"Prediction: {response['prediction']} ({probability:.2%} churn probability)")
        
        return jsonify(response)
        
    except ValueError as e:
        logger.warning(f"Validation error: {e}")
        return jsonify({'error': str(e), 'valid_values': VALID_VALUES}), 400
    except KeyError as e:
        logger.warning(f"Missing field: {e}")
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting E-Commerce Churn Prediction API...")
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )