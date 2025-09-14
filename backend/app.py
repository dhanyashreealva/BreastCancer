from flask import Flask, request, jsonify
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__)

# Load model & scaler
model = load_model("breast_cancer_model.h5")
scaler = joblib.load("scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data['features']).reshape(1, -1)
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0][0]
        result = "Benign ✅" if prediction > 0.5 else "Malignant ⚠️"

        return jsonify({
            "prediction": result,
            "confidence": float(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
