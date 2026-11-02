
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Model file location
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"

# Required prediction features
FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    """Load the trained student result prediction model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# Health check endpoint
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# Student result prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "JSON request body is required",
            "missing_fields": FEATURES
        }), 400

    # Check for missing fields
    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    # Validate numeric input
    try:
        values = {
            feature: float(data[feature])
            for feature in FEATURES
        }
    except (TypeError, ValueError):
        return jsonify({
            "error": "All input fields must contain numeric values"
        }), 400

    # Validate score ranges
    for feature, value in values.items():
        if not 0 <= value <= 100:
            return jsonify({
                "error": f"{feature} must be between 0 and 100"
            }), 400

    try:
        # Prepare input for the ML model
        sample = pd.DataFrame([values], columns=FEATURES)

        model = load_model()

        # Predict student result
        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error)
        }), 500

    except Exception as error:
        return jsonify({
            "error": f"Prediction failed: {str(error)}"
        }), 500


# Run Flask application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
