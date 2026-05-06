from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Home route
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

    # Safe input handling
    try:
        values = [data.get(f, 0) for f in features]
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    # Prediction
    prediction = model.predict([values])[0]

    # Probability (safe check)
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba([values])[0][1]
    else:
        probability = 0.5

    # Confidence level
    if probability > 0.8:
        confidence = "High"
    elif probability > 0.6:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Feature importance (safe check)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        top_index = np.argsort(importances)[-3:][::-1]
        top_factors = [features[i] for i in top_index]
    else:
        top_factors = features[:3]

    return jsonify({
        "prediction": "High Success" if prediction == 1 else "Low Success",
        "probability": float(probability),
        "confidence": confidence,
        "top_factors": top_factors
    })


# Chat route (FIXED - outside predict)
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json

    question = data.get("question", "")
    prediction = data.get("prediction", "unknown")
    factors = data.get("factors", [])

    if "why" in question.lower():
        reply = f"The success is {prediction.lower()} mainly due to {', '.join(factors)}."
    elif "improve" in question.lower():
        reply = "Improving bone density and avoiding risk factors like smoking can increase success."
    else:
        reply = "This prediction is based on clinical factors like bone condition and patient health."

    return jsonify({"reply": reply})


if __name__ == "__main__":
    app.run(debug=True)