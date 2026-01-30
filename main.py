"""
Flask API for Customer Churn Prediction
"""

from flask import Flask, request, jsonify
from predict import predict_churn
import os

app = Flask(__name__)

# Map numeric output to human-readable labels
label_map = {
    0: "No Churn",
    1: "Churn"
}

@app.route('/')
def home():
    return jsonify({
        'message': 'Customer Churn Prediction API',
        'endpoints': {
            '/predict': 'POST - Make a churn prediction',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Extract features from request
        tenure = int(data['tenure'])
        monthly_charges = float(data['monthly_charges'])
        total_charges = float(data['total_charges'])
        contract = data['contract']
        internet_service = data['internet_service']
        payment_method = data['payment_method']
        
        print(f"Input: tenure={tenure}, monthly_charges={monthly_charges}, "
              f"total_charges={total_charges}, contract={contract}, "
              f"internet_service={internet_service}, payment_method={payment_method}")
        
        # Call model
        prediction, probability = predict_churn(
            tenure, monthly_charges, total_charges,
            contract, internet_service, payment_method
        )
        
        # Get label
        pred_label = label_map.get(prediction, str(prediction))
        
        return jsonify({
            'prediction': pred_label,
            'churn_probability': round(probability, 4),
            'retention_probability': round(1 - probability, 4)
        })
        
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8080))
    )
