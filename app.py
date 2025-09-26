from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# --- STEP 1: Load the Model Once ---
try:
    with open('xgb_model.pkl', 'rb') as f:
        GLOBAL_MODEL = pickle.load(f)
except Exception as e:
    print(f"Error loading model: {e}")
    GLOBAL_MODEL = None

    # --- 2. ADD THE NEW STATUS ENDPOINT HERE ---
@app.route("/", methods=["GET"])
def home():
    # This is a good way to confirm the server is running
    return jsonify({
        "service_name": "ML Forecast API",
        "status": "Server is running, but check /predict for the ML model."
    }), 200


# --- STEP 3: Define the Route ---
@app.route("/predict", methods=["POST"])
def predict_demand():
    if GLOBAL_MODEL is None:
        return jsonify({"error": "Model failed to load"}), 500
    
    try:
        # --- STEP 4: Get Input Data ---
        data = request.get_json()
        input_data = data.get('features') # Expecting a list like [2025, 10, 150, 180, ...]
        
        if not input_data:
             return jsonify({"error": "Missing 'features' in request body"}), 400

        # --- STEP 5: Prepare Data ---
        features_array = np.array(input_data).reshape(1, -1) 

        # --- STEP 6: Run the Prediction ---
        raw_prediction = GLOBAL_MODEL.predict(features_array)
        predicted_demand = int(round(raw_prediction[0]))
        
        # --- STEP 7: Return JSON Response ---
        return jsonify({
            "status": "success",
            "predicted_demand": predicted_demand
        }), 200

    except Exception as e:
        return jsonify({"error": f"An error occurred during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    # Render will use the Gunicorn/production server, but this is for local testing
    app.run(debug=True)
