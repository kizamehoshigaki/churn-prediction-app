"""
Streamlit Frontend for E-Commerce Customer Churn Prediction
Beautiful, interactive dashboard for business users
"""

import streamlit as st
import requests
import json

# UPDATE THIS URL after deploying to Cloud Run
API_URL = "https://ecommerce-churn-app-957487780317.us-central1.run.app"  # your actual URL"

st.set_page_config(
    page_title="E-Commerce Churn Predictor",
    page_icon="🛒",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .risk-critical { color: #d62728; font-weight: bold; }
    .risk-high { color: #ff7f0e; font-weight: bold; }
    .risk-medium { color: #ffbb00; font-weight: bold; }
    .risk-low { color: #2ca02c; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🛒 E-Commerce Customer Churn Prediction")
st.markdown("Predict which customers are likely to churn and get actionable retention recommendations.")

st.divider()

# Create tabs
tab1, tab2 = st.tabs(["🔮 Predict Churn", "📊 About Model"])

with tab1:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("👤 Customer Profile")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        gender = st.selectbox("Gender", ["Male", "Female"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        city_tier = st.selectbox("City Tier", [1, 2, 3])
        number_of_address = st.number_input("Number of Addresses", 1, 20, 3)
    
    with col2:
        st.subheader("📱 Engagement")
        preferred_login_device = st.selectbox("Preferred Login Device", 
                                               ["Mobile Phone", "Computer", "Phone"])
        hour_spend_on_app = st.slider("Hours Spent on App", 0.0, 10.0, 3.0, 0.5)
        number_of_device_registered = st.slider("Devices Registered", 1, 10, 3)
        prefered_order_cat = st.selectbox("Preferred Category", 
                                          ["Laptop & Accessory", "Mobile Phone", "Fashion", "Grocery", "Others"])
        day_since_last_order = st.slider("Days Since Last Order", 0, 60, 5)
    
    with col3:
        st.subheader("💰 Transaction")
        order_count = st.number_input("Total Orders", 1, 100, 5)
        cashback_amount = st.number_input("Cashback Amount ($)", 0.0, 500.0, 150.0)
        coupon_used = st.number_input("Coupons Used", 0, 20, 2)
        order_amount_hike = st.slider("Order Amount Hike (%)", 0, 30, 15)
        preferred_payment_mode = st.selectbox("Payment Method", 
                                               ["Credit Card", "Debit Card", "UPI", "Cash on Delivery", "E wallet"])

    st.divider()
    
    col_sat, col_comp, col_dist = st.columns(3)
    with col_sat:
        satisfaction_score = st.slider("⭐ Satisfaction Score", 1, 5, 3)
    with col_comp:
        complain = st.selectbox("⚠️ Has Complained?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0])[1]
    with col_dist:
        warehouse_to_home = st.number_input("🚚 Distance to Warehouse (km)", 1, 100, 15)

    st.divider()
    
    # Predict button
    if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
        payload = {
            "tenure": tenure,
            "warehouse_to_home": warehouse_to_home,
            "hour_spend_on_app": hour_spend_on_app,
            "number_of_device_registered": number_of_device_registered,
            "satisfaction_score": satisfaction_score,
            "number_of_address": number_of_address,
            "complain": complain,
            "order_amount_hike": order_amount_hike,
            "coupon_used": coupon_used,
            "order_count": order_count,
            "day_since_last_order": day_since_last_order,
            "cashback_amount": cashback_amount,
            "city_tier": city_tier,
            "preferred_login_device": preferred_login_device,
            "preferred_payment_mode": preferred_payment_mode,
            "gender": gender,
            "prefered_order_cat": prefered_order_cat,
            "marital_status": marital_status
        }
        
        try:
            with st.spinner("Analyzing customer data..."):
                response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
                result = response.json()
            
            if response.status_code == 200:
                st.divider()
                
                # Results
                col_pred, col_prob, col_risk = st.columns(3)
                
                with col_pred:
                    if result['prediction'] == "Churn":
                        st.error(f"### ⚠️ {result['prediction']}")
                    else:
                        st.success(f"### ✅ {result['prediction']}")
                
                with col_prob:
                    st.metric("Churn Probability", f"{result['churn_probability']*100:.1f}%")
                
                with col_risk:
                    risk = result['risk_level']
                    color_map = {'Critical': '🔴', 'High': '🟠', 'Medium': '🟡', 'Low': '🟢'}
                    st.metric("Risk Level", f"{color_map.get(risk, '')} {risk}")
                
                # Progress bar
                st.progress(result['churn_probability'], text=f"Churn Risk: {result['churn_probability']*100:.1f}%")
                
                # Risk factors and recommendations
                col_risks, col_recs = st.columns(2)
                
                with col_risks:
                    st.subheader("⚠️ Risk Factors")
                    for factor in result['risk_factors']:
                        st.warning(factor)
                
                with col_recs:
                    st.subheader("💡 Recommendations")
                    for rec in result['recommendations']:
                        st.info(rec)
                
            else:
                st.error(f"Error: {result.get('error', 'Unknown error')}")
                
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Could not connect to API. Make sure the Flask server is running.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.subheader("📊 About This Model")
    
    st.markdown("""
    ### Model Overview
    This E-Commerce Customer Churn Prediction system uses **XGBoost** with **SMOTE** 
    (Synthetic Minority Over-sampling Technique) to handle class imbalance.
    
    ### Dataset
    - **Source:** E-Commerce Customer Churn Dataset (Kaggle)
    - **Samples:** ~5,600 customers
    - **Features:** 18 customer attributes
    - **Churn Rate:** ~17%
    
    ### Key Predictive Features
    1. **Tenure** - New customers (<6 months) have highest churn risk
    2. **Complaints** - Customers who complained are 3x more likely to churn
    3. **Satisfaction Score** - Strong negative correlation with churn
    4. **Days Since Last Order** - Inactivity signals disengagement
    5. **Cashback Amount** - Higher engagement = lower churn
    
    ### Model Performance
    | Metric | Score |
    |--------|-------|
    | Accuracy | ~92% |
    | Precision | ~88% |
    | Recall | ~75% |
    | F1-Score | ~81% |
    | ROC-AUC | ~94% |
    
    ### Business Impact
    Using this model, the e-commerce company can:
    - Identify at-risk customers before they churn
    - Prioritize retention efforts on high-value customers
    - Reduce customer acquisition costs by improving retention
    - Increase customer lifetime value (CLV)
    """)

# Sidebar
with st.sidebar:
    st.header("🛒 E-Commerce Churn")
    st.markdown("""
    This app predicts customer churn for e-commerce businesses using machine learning.
    
    **How to use:**
    1. Enter customer information
    2. Click 'Predict Churn Risk'
    3. Review risk factors & recommendations
    
    ---
    
    **Built with:**
    - XGBoost
    - SMOTE
    - Flask API
    - Streamlit
    
    ---
    

    Northeastern University
    """)
    
    st.divider()
    
    # API Status
    try:
        health = requests.get(f"{API_URL}/health", timeout=5)
        if health.status_code == 200:
            st.success("✅ API Status: Online")
        else:
            st.error("❌ API Status: Error")
    except:
        st.warning("⚠️ API Status: Offline")