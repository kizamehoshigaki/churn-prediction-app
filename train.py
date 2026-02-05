"""
E-Commerce Customer Churn Model Training Pipeline
- Compares multiple ML models
- Handles class imbalance with class weights
- Performs hyperparameter tuning
- Saves best model with metadata
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, roc_auc_score)
import xgboost as xgb
import pickle
import os
import json
from datetime import datetime

# Create directories
os.makedirs('model', exist_ok=True)
os.makedirs('images', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("=" * 70)
print("   E-COMMERCE CUSTOMER CHURN - MODEL TRAINING PIPELINE")
print("=" * 70)

# =============================================================================
# 1. DATA LOADING
# =============================================================================
print("\n[1/5] Loading and exploring data...")

try:
    df = pd.read_csv('data/ecommerce_churn.csv')
    print("   ✓ Loaded from: data/ecommerce_churn.xlsx")
except FileNotFoundError:
    print("   ⚠ File not found. Please download the dataset from:")
    print("     https://www.kaggle.com/datasets/ankitverma2010/ecommerce-customer-churn-analysis-and-prediction")
    print("     Save as: data/ecommerce_churn.csv")
    raise FileNotFoundError("Dataset not found!")

print(f"   Dataset shape: {df.shape}")

# =============================================================================
# 2. DATA PREPROCESSING
# =============================================================================
print("\n[2/5] Preprocessing data...")

# Drop CustomerID if exists
if 'CustomerID' in df.columns:
    df = df.drop('CustomerID', axis=1)

# Handle missing values
print(f"   Missing values before: {df.isnull().sum().sum()}")
df = df.dropna()
df = df.drop_duplicates()
print(f"   Dataset shape after cleaning: {df.shape}")

# Check target distribution
churn_rate = df['Churn'].mean() * 100
print(f"   Churn rate: {churn_rate:.1f}%")

# Define feature columns (check which ones exist in dataset)
possible_numerical = ['Tenure', 'WarehouseToHome', 'HourSpendOnApp', 
                     'NumberOfDeviceRegistered', 'SatisfactionScore', 'NumberOfAddress',
                     'Complain', 'OrderAmountHikeFromlastYear', 'CouponUsed', 
                     'OrderCount', 'DaySinceLastOrder', 'CashbackAmount', 'CityTier']

possible_categorical = ['PreferredLoginDevice', 'PreferredPaymentMode', 'Gender', 
                       'PreferedOrderCat', 'MaritalStatus']

# Filter to columns that actually exist
numerical_cols = [c for c in possible_numerical if c in df.columns]
categorical_cols = [c for c in possible_categorical if c in df.columns]

print(f"   Numerical features: {len(numerical_cols)}")
print(f"   Categorical features: {len(categorical_cols)}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

features = numerical_cols + categorical_cols
print(f"   Total features: {len(features)}")

# Prepare X and y
X = df[features]
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"   Training set: {X_train.shape[0]} samples")
print(f"   Test set: {X_test.shape[0]} samples")

# Calculate class weight for imbalanced data
class_counts = y_train.value_counts()
scale_pos_weight = class_counts[0] / class_counts[1]
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}")

# =============================================================================
# 3. MODEL COMPARISON
# =============================================================================
print("\n[3/5] Comparing multiple models...")

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, random_state=42, 
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss', verbosity=0
    )
}

results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1-Score': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    results.append(metrics)
    print(f"   {name}: F1={metrics['F1-Score']:.3f}, ROC-AUC={metrics['ROC-AUC']:.3f}")

# Save comparison
results_df = pd.DataFrame(results)
results_df.to_csv('model/model_comparison.csv', index=False)

best_model_name = results_df.loc[results_df['ROC-AUC'].idxmax(), 'Model']
print(f"\n   🏆 Best model: {best_model_name}")

# =============================================================================
# 4. HYPERPARAMETER TUNING (XGBoost)
# =============================================================================
print("\n[4/5] Tuning XGBoost hyperparameters...")

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 7],
    'learning_rate': [0.1, 0.2],
}

xgb_model = xgb.XGBClassifier(
    random_state=42, 
    scale_pos_weight=scale_pos_weight,
    eval_metric='logloss', 
    verbosity=0
)

grid_search = GridSearchCV(
    xgb_model, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0
)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
print(f"   Best parameters: {grid_search.best_params_}")
print(f"   Best CV ROC-AUC: {grid_search.best_score_:.4f}")

# =============================================================================
# 5. FINAL EVALUATION & SAVE
# =============================================================================
print("\n[5/5] Final evaluation and saving model...")

y_pred_final = best_model.predict(X_test)
y_prob_final = best_model.predict_proba(X_test)[:, 1]

final_metrics = {
    'accuracy': float(accuracy_score(y_test, y_pred_final)),
    'precision': float(precision_score(y_test, y_pred_final)),
    'recall': float(recall_score(y_test, y_pred_final)),
    'f1_score': float(f1_score(y_test, y_pred_final)),
    'roc_auc': float(roc_auc_score(y_test, y_prob_final))
}

print(f"\n   Final Model Performance:")
print(f"   {'='*40}")
print(f"   Accuracy:  {final_metrics['accuracy']:.4f}")
print(f"   Precision: {final_metrics['precision']:.4f}")
print(f"   Recall:    {final_metrics['recall']:.4f}")
print(f"   F1-Score:  {final_metrics['f1_score']:.4f}")
print(f"   ROC-AUC:   {final_metrics['roc_auc']:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 5 Most Important Features:")
for idx, row in feature_importance.head(5).iterrows():
    print(f"   - {row['feature']}: {row['importance']:.4f}")

# Save model artifacts
model_artifacts = {
    'model': best_model,
    'label_encoders': label_encoders,
    'feature_names': features,
    'categorical_cols': categorical_cols,
    'numerical_cols': numerical_cols,
    'metrics': final_metrics,
    'best_params': grid_search.best_params_,
    'training_date': datetime.now().isoformat()
}

with open('model/churn_model.pkl', 'wb') as f:
    pickle.dump(model_artifacts, f)

# Save metadata as JSON
metadata = {
    'model_type': 'XGBoost',
    'dataset': 'E-Commerce Customer Churn',
    'samples_trained': int(len(X_train)),
    'samples_tested': int(len(X_test)),
    'features_count': len(features),
    'metrics': final_metrics,
    'best_params': grid_search.best_params_,
    'feature_importance': feature_importance.head(10).to_dict('records'),
    'training_date': datetime.now().isoformat(),
    'churn_rate': float(churn_rate)
}

with open('model/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

# Save feature importance
feature_importance.to_csv('model/feature_importance.csv', index=False)

print("\n" + "=" * 70)
print("   ✅ TRAINING COMPLETE!")
print("=" * 70)
print(f"\n   Files saved:")
print(f"   - model/churn_model.pkl")
print(f"   - model/model_metadata.json")
print(f"   - model/model_comparison.csv")
print(f"   - model/feature_importance.csv")

# Print encoding reference
print("\n" + "=" * 70)
print("   ENCODING REFERENCE")
print("=" * 70)
for col, le in label_encoders.items():
    print(f"\n   {col}:")
    for i, label in enumerate(le.classes_):
        print(f"      {label} -> {i}")