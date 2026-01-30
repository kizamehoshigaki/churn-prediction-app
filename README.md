# Customer Churn Prediction API

A Flask-based REST API for predicting customer churn, deployed on Google Cloud Platform.

## Overview

This project demonstrates MLOps practices by building and deploying a machine learning model that predicts whether a telecom customer will churn (leave the company) based on their account information.

**Model:** Random Forest Classifier  
**Dataset:** [Telco Customer Churn Dataset](https://github.com/IBM/telco-customer-churn-on-icp4d)

## Features Used

| Feature | Description |
|---------|-------------|
| `tenure` | Number of months with the company |
| `monthly_charges` | Monthly bill amount |
| `total_charges` | Total amount charged |
| `contract` | Contract type (Month-to-month, One year, Two year) |
| `internet_service` | Internet service type (DSL, Fiber optic, No) |
| `payment_method` | Payment method |

## Project Structure

```
├── model/
│   └── churn_model.pkl    # Trained model
├── main.py                # Flask API
├── predict.py             # Prediction module
├── streamlit_app.py       # Frontend application
├── train_model.py         # Model training script
├── Dockerfile
├── requirements.txt
└── README.md
```

## Setup & Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_model.py
```

### 3. Run Locally
```bash
python main.py
```
The API will be available at `http://localhost:8080`

### 4. Run Streamlit Frontend
```bash
streamlit run streamlit_app.py
```

## API Endpoints

### Health Check
```
GET /health
```

### Predict Churn
```
POST /predict
Content-Type: application/json

{
    "tenure": 12,
    "monthly_charges": 50.0,
    "total_charges": 600.0,
    "contract": "Month-to-month",
    "internet_service": "Fiber optic",
    "payment_method": "Electronic check"
}
```

**Response:**
```json
{
    "prediction": "Churn",
    "churn_probability": 0.73,
    "retention_probability": 0.27
}
```

## GCP Deployment

### Prerequisites
- [Google Cloud CLI](https://cloud.google.com/sdk/docs/install)
- GCP Project with billing enabled
- Artifact Registry and Cloud Build APIs enabled

### Deploy Steps

1. **Authenticate with GCP:**
```bash
gcloud init
gcloud auth login
```

2. **Enable required APIs:**
```bash
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable artifactregistry.googleapis.com
```

3. **Build and push container:**
```bash
gcloud builds submit --tag gcr.io/mlops-churn-lab/churn-app
```

4. **Deploy to Cloud Run:**
```bash
gcloud run deploy churn-app --image gcr.io/mlops-churn-lab/churn-app --platform managed --port 8080 --allow-unauthenticated
```

5. **Update Streamlit app** with the deployed URL and run:
```bash
streamlit run streamlit_app.py
```

## Author

Aaditya - IE7374 MLOps Lab Assignment  
Northeastern University