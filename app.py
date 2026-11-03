
from pathlib import Path
import logging
import sys

import joblib
import pandas as pd
from flask import Flask, jsonify, request

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger("student-result-api")

logger.info("Student Result Prediction API is starting")

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
        logger.error("Model file not found: %s", MODEL_PATH)
        raise FileNotFoundError(
            "Model file not found. Please run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


# Health check endpoint
@app.route("/", methods=["GET"])
@app.route("/health", methods=["GET"])
def health_check():
    logger.info("Health check successful")

    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# Student result prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():

    logger.info("Prediction request received")

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning("Prediction failed: JSON request body is missing")

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
        logger.warning(
            "Prediction failed: Missing fields: %s",
            missing_fields
        )

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
        logger.warning("Prediction failed: Non-numeric input")

        return jsonify({
            "error": "All input fields must contain numeric values"
        }), 400

    # Validate score ranges
    for feature, value in values.items():

        if not 0 <= value <= 100:
            logger.warning(
                "Prediction failed: Invalid range for %s",
                feature
            )

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

        # Log successful prediction
        logger.info(
            "Prediction successful | Result: %s | Prediction code: %s",
            prediction,
            prediction_code
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except FileNotFoundError as error:
        logger.exception("Prediction failed: Model file error")

        return jsonify({
            "error": str(error)
        }), 500

    except Exception as error:
        logger.exception("Prediction failed due to an unexpected error")

        return jsonify({
            "error": f"Prediction failed: {str(error)}"
        }), 500


# Run Flask application
if __name__ == "__main__":
    logger.info("Starting Flask server on port 5000")

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
