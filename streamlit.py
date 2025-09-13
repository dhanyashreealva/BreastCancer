import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sklearn.datasets import load_breast_cancer

# Load model & scaler
model = load_model("breast_cancer_model.h5")
scaler = joblib.load("scaler.pkl")

# Load feature names
data = load_breast_cancer()
feature_names = data.feature_names

st.title("🩺 Breast Cancer Prediction (ANN Model)")
st.write("Enter tumor measurements below to predict if it is **benign or malignant**.")

# User inputs
user_input = []
for feature in feature_names:
    val = st.number_input(f"{feature}", value=float(np.mean(data.data[:, list(feature_names).tolist().index(feature)])))
    user_input.append(val)

# Convert to array and scale
input_data = np.array(user_input).reshape(1, -1)
input_scaled = scaler.transform(input_data)

# Predict
if st.button("🔍 Predict"):
    prediction = model.predict(input_scaled)[0][0]
    result = "Benign ✅" if prediction > 0.5 else "Malignant ⚠️"
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {prediction:.4f}")
