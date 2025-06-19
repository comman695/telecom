
import streamlit as st
import joblib
import pandas as pd

# Load model and preprocessing tools
model = joblib.load("final_xgboost_telco_model.pkl")
scaler = joblib.load("preprocessing_pipeline.pkl")
label_encoder = joblib.load("label_encoder_plan_type.pkl")

st.title("Telco Customer Churn Prediction")

# Collect user input
plan_type = st.selectbox("Plan Type", label_encoder.classes_)
international_plan = st.selectbox("International Plan", [0, 1])
voice_mail_plan = st.selectbox("Voice Mail Plan", [0, 1])
number_vmail_messages = st.number_input("Number of Voicemail Messages", 0, 100, 0)
total_day_minutes = st.number_input("Total Day Minutes", 0.0, 500.0, 200.0)
total_eve_minutes = st.number_input("Total Evening Minutes", 0.0, 500.0, 200.0)
total_night_minutes = st.number_input("Total Night Minutes", 0.0, 500.0, 200.0)
total_intl_minutes = st.number_input("Total International Minutes", 0.0, 100.0, 10.0)
customer_service_calls = st.slider("Customer Service Calls", 0, 10, 1)

# Convert inputs to dataframe
input_data = pd.DataFrame([{
    "plan_type": label_encoder.transform([plan_type])[0],
    "international_plan": international_plan,
    "voice_mail_plan": voice_mail_plan,
    "number_vmail_messages": number_vmail_messages,
    "total_day_minutes": total_day_minutes,
    "total_eve_minutes": total_eve_minutes,
    "total_night_minutes": total_night_minutes,
    "total_intl_minutes": total_intl_minutes,
    "customer_service_calls": customer_service_calls
}])

# Scale input
scaled_data = scaler.transform(input_data)

# Predict
prediction = model.predict(scaled_data)[0]
st.subheader("Prediction Result")
st.write("Churn" if prediction == 1 else "No Churn")
