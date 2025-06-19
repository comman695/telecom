
import streamlit as st
import pandas as pd
import joblib

# Load saved components
model = joblib.load("final_xgboost_telco_model.pkl")
scaler = joblib.load("preprocessing_pipeline.pkl")
label_encoder = joblib.load("label_encoder_plan_type.pkl")

st.title("Telco Customer Churn Prediction")

# Define input fields
intl = st.number_input("Intl Charge", min_value=0.0)
vmail = st.number_input("Voicemail Message Count", min_value=0.0)
day = st.number_input("Day Charge", min_value=0.0)
eve = st.number_input("Evening Charge", min_value=0.0)
night = st.number_input("Night Charge", min_value=0.0)
calls = st.number_input("Customer Service Calls", min_value=0)
plan_type = st.selectbox("Plan Type", label_encoder.classes_)

if st.button("Predict"):
    # Prepare input data
    input_dict = {
        "intl": intl,
        "vmail": vmail,
        "day": day,
        "eve": eve,
        "night": night,
        "calls": calls,
        "plan_type": plan_type
    }
    input_df = pd.DataFrame([input_dict])

    # Encode categorical
    input_df['plan_type'] = label_encoder.transform(input_df['plan_type'])

    # Ensure correct feature order
    input_df = input_df[scaler.feature_names_in_]

    # Transform and predict
    X_scaled = scaler.transform(input_df)
    prediction = model.predict(X_scaled)[0]
    result = "Churn" if prediction == 1 else "Not Churn"
    st.success(f"Prediction: {result}")
