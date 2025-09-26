from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model globally on startup
MODEL_PATH = 'xgb_model.pkl'
try:
    if os.path.exists(MODEL_PATH):
        # We use joblib as per your original file
        GLOBAL_MODEL = joblib.load(MODEL_PATH)
    else:
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    GLOBAL_MODEL = None


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Safety Check
    if GLOBAL_MODEL is None:
        return jsonify({"status": "error", "message": "Model not loaded on server."}), 500

    # 2. Correctly read JSON data (this fixes the non-changing output bug)
    try:
        data = request.get_json(force=True)
        # Ensure 'features' key exists and is a list
        features_list = data.get('features')

        # 3. Check for 5 features (CRITICAL)
        if features_list is None or not isinstance(features_list, list) or len(features_list) != 5:
            return jsonify({
                "status": "error",
                "message": f"Invalid input. Expected exactly 5 features, received {len(features_list) if features_list else 0}."
            }), 400

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error processing input data: {e}"
        }), 400
    
    # 4. Convert and Predict
    try:
        final_features = np.array([features_list])
        prediction = GLOBAL_MODEL.predict(final_features)
        
        # Convert prediction to standard float/int for JSON
        predicted_demand = float(prediction[0])
        
        return jsonify({
            "status": "success",
            "predicted_demand": round(predicted_demand) # Return as a whole number
        }), 200

    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Error during prediction: {e}"
        }), 500


@app.route('/')
def home():
    return jsonify({
        "service_name": "ML Forecast API",
        "status": "Server is running, but check /predict for the ML model (POST with 5 features)."
    }), 200


if __name__ == '__main__':
    # Use 0.0.0.0 for external access in production environments like Render
    port = int(os.environ.get("PORT", 5000)) 
    app.run(host='0.0.0.0', port=port)
