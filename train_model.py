"""
Customer Churn Model Training Script
Run this once to generate the churn_model.pkl file
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Create model folder if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load the Telco Customer Churn dataset
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)

# Select features we'll use for prediction
features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'PaymentMethod']
target = 'Churn'

# Clean data
df = df[features + [target]].copy()
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df = df.dropna()

# Encode categorical variables
label_encoders = {}
categorical_cols = ['Contract', 'InternetService', 'PaymentMethod']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

# Split features and target
X = df[features]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Save model and encoders
model_artifacts = {
    'model': model,
    'label_encoders': label_encoders,
    'feature_names': features
}

with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

print("\nModel saved to model/churn_model.pkl")

# Print encoding mappings for reference
print("\n--- Encoding Reference ---")
for col, le in label_encoders.items():
    print(f"\n{col}:")
    for i, label in enumerate(le.classes_):
        print(f"  {label} -> {i}")