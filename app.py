import streamlit as st
import pandas as pd
import pickle

st.title("Mental Health Stress Prediction")

# Load model & scaler
model = pickle.load(open("artifacts/models/xgboost.pkl", "rb"))
scaler = pickle.load(open("artifacts/models/scaler.pkl", "rb"))

st.write("Enter feature values:")

# Example input (replace with your real feature names)
input_data = {}
for col in ["feature1", "feature2", "feature3"]:  
    input_data[col] = st.number_input(col)

if st.button("Predict"):
    df = pd.DataFrame([input_data])
    df_scaled = scaler.transform(df)
    prediction = model.predict(df_scaled)[0]
    st.success(f"Predicted Stress Level: {prediction + 1}")