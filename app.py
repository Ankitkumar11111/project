from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load trained model
model = joblib.load("xgb_model.pkl")

app = Flask(__name__)

@app.route('/')
def home():
    return "✅ ML API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)
        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
