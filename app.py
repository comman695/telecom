
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('xgboost_best_model.pkl')

st.set_page_config(page_title="Customer Churn Predictor", layout="centered")

st.title("📊 Customer Churn Prediction App")
st.markdown("Enter customer details to predict churn.")

# Input form
with st.form("churn_form"):
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    monthly_charges = st.slider("Monthly Charges ($)", 10.0, 150.0, 70.0)

    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])

    submitted = st.form_submit_button("Predict")

# Make prediction
if submitted:
    input_df = pd.DataFrame([{
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'Contract': contract,
        'InternetService': internet_service,
        'PaymentMethod': payment_method
    }])

    prediction = model.predict(input_df)[0]
    churn_prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ This customer is likely to churn. (Probability: {churn_prob:.2f})")
    else:
        st.success(f"✅ This customer is likely to stay. (Probability: {1 - churn_prob:.2f})")
