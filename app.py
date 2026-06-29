import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ---------------- PAGE ----------------

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("📊 Customer Churn Prediction")
st.write("Fill customer details and click Predict.")

# ---------------- LOAD FILES ----------------

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# ---------------- INPUTS ----------------

gender = st.selectbox("Gender",["Male","Female"])

senior = st.selectbox("Senior Citizen",[0,1])

partner = st.selectbox("Partner",["Yes","No"])

dependents = st.selectbox("Dependents",["Yes","No"])

phone = st.selectbox("Phone Service",["Yes","No"])

multiple = st.selectbox(
    "Multiple Lines",
    ["No","Yes","No phone service"]
)

internet = st.selectbox(
    "Internet Service",
    ["DSL","Fiber optic","No"]
)

online_security = st.selectbox(
    "Online Security",
    ["Yes","No","No internet service"]
)

online_backup = st.selectbox(
    "Online Backup",
    ["Yes","No","No internet service"]
)

device = st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
)

tech = st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
)

tv = st.selectbox(
    "Streaming TV",
    ["Yes","No","No internet service"]
)

movies = st.selectbox(
    "Streaming Movies",
    ["Yes","No","No internet service"]
)

contract = st.selectbox(
    "Contract",
    ["Month-to-month","One year","Two year"]
)

paper = st.selectbox(
    "Paperless Billing",
    ["Yes","No"]
)

payment = st.selectbox(
    "Payment Method",
    [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ]
)

tenure = st.slider("Tenure",0,72,12)

monthly = st.number_input(
    "Monthly Charges",
    0.0,
    150.0,
    70.0
)

total = st.number_input(
    "Total Charges",
    0.0,
    10000.0,
    1000.0
)

predict = st.button("Predict Churn")

if predict:

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "PhoneService": phone,
        "MultipleLines": multiple,
        "InternetService": internet,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device,
        "TechSupport": tech,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paper,
        "PaymentMethod": payment,
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total
    }

    df = pd.DataFrame([data])
    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    df_scaled = scaler.transform(df)

    prediction = model.predict(df_scaled)[0]
    probability = model.predict_proba(df_scaled)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("❌ Customer will Churn")
    else:
        st.success("✅ Customer will NOT Churn")

    st.write(f"Probability : {probability:.2%}")
