"""
Streamlit Frontend for Customer Churn Prediction
"""

import streamlit as st
import requests

# UPDATE THIS URL after deploying to Cloud Run
API_URL = " https://churn-app-957487780317.us-east1.run.app"  # Change to your Cloud Run URL after deployment

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊")

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a customer is likely to churn based on their account information.")

st.divider()

# Input form
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)", 
        min_value=0, 
        max_value=72, 
        value=12,
        help="Number of months the customer has been with the company"
    )
    
    monthly_charges = st.number_input(
        "Monthly Charges ($)", 
        min_value=0.0, 
        max_value=200.0, 
        value=50.0,
        help="Monthly bill amount"
    )
    
    total_charges = st.number_input(
        "Total Charges ($)", 
        min_value=0.0, 
        max_value=10000.0, 
        value=600.0,
        help="Total amount charged to the customer"
    )

with col2:
    contract = st.selectbox(
        "Contract Type",
        options=["Month-to-month", "One year", "Two year"],
        help="Type of contract the customer has"
    )
    
    internet_service = st.selectbox(
        "Internet Service",
        options=["DSL", "Fiber optic", "No"],
        help="Customer's internet service type"
    )
    
    payment_method = st.selectbox(
        "Payment Method",
        options=[
            "Electronic check",
            "Mailed check", 
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ],
        help="How the customer pays their bill"
    )

st.divider()

# Predict button
if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
    # Prepare payload
    payload = {
        "tenure": tenure,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "contract": contract,
        "internet_service": internet_service,
        "payment_method": payment_method
    }
    
    try:
        # Make API request
        response = requests.post(f"{API_URL}/predict", json=payload)
        result = response.json()
        
        if response.status_code == 200:
            prediction = result['prediction']
            churn_prob = result['churn_probability']
            
            st.divider()
            
            # Display result with color coding
            if prediction == "Churn":
                st.error(f"⚠️ **Prediction: {prediction}**")
                st.metric("Churn Probability", f"{churn_prob * 100:.1f}%")
                st.warning("This customer is at risk of leaving. Consider retention strategies!")
            else:
                st.success(f"✅ **Prediction: {prediction}**")
                st.metric("Retention Probability", f"{(1-churn_prob) * 100:.1f}%")
                st.info("This customer is likely to stay. Keep up the good service!")
                
            # Show probability bar
            st.progress(churn_prob, text=f"Churn Risk: {churn_prob*100:.1f}%")
            
        else:
            st.error(f"Error: {result.get('error', 'Unknown error')}")
            
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Could not connect to the API. Make sure the Flask server is running.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Sidebar info
with st.sidebar:
    st.header("About")
    st.markdown("""
    This app predicts customer churn using a Random Forest model 
    trained on the Telco Customer Churn dataset.
    
    **Features used:**
    - Tenure
    - Monthly Charges
    - Total Charges
    - Contract Type
    - Internet Service
    - Payment Method
    
    **Model:** Random Forest Classifier
    """)
    
    st.divider()
    st.markdown("Built for IE7374 MLOps Lab Assignment")