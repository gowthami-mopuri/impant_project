from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Home route (to check API is running)
@app.route('/')
def home():
    return "API is running"

# Load model
bundle = pickle.load(open("model.pkl", "rb"))
model = bundle["model"]
features = bundle["features"]

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    # Arrange input values in correct order
    values = [data[f] for f in features]

    # Prediction
    prediction = model.predict([values])[0]
    probability = model.predict_proba([values])[0][1]

    # Confidence level
    if probability > 0.8:
        confidence = "High"
    elif probability > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Top factors (feature importance)
    importances = model.feature_importances_
    top_index = np.argsort(importances)[-3:][::-1]
    top_factors = [features[i] for i in top_index]

    # Final response
    return jsonify({
        "prediction": "High Success" if prediction == 1 else "Low Success",
        "probability": float(probability),
        "confidence": confidence,
        "top_factors": top_factors
    })

if __name__ == "__main__":
    app.run(debug=True)