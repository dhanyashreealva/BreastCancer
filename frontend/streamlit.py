import streamlit as st
import requests
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load feature names
data = load_breast_cancer()
feature_names = list(data.feature_names)

st.title("Breast Cancer Prediction")
st.write("Enter tumor measurements below to predict if it is **benign or malignant**.")

# User inputs
user_input = []
for i, feature in enumerate(feature_names):
    val = st.number_input(f"{feature}", value=float(np.mean(data.data[:, i])))
    user_input.append(val)

# Send to backend API
if st.button("Predict"):
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"features": user_input}
        )
        result = response.json()
        if "error" in result:
            st.error(f"Error: {result['error']}")
        else:
            st.subheader(f"Prediction: {result['prediction']}")
            st.write(f"Confidence: {result['confidence']:.4f}")
    except Exception as e:
        st.error(f"Could not connect to backend: {e}")
