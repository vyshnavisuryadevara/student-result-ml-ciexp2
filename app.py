
import os
import sys
import time
import uuid
import logging
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g, Response

from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    generate_latest,
    CONTENT_TYPE_LATEST
)


# STEP 1: CONFIGURE LOGGING

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")


# STEP 2: CREATE FLASK APPLICATION

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


# STEP 3: CONFIGURE PROMETHEUS METRICS

# Counts HTTP requests by method, endpoint and response status.
HTTP_REQUESTS = Counter(
    "http_requests",
    "Total number of HTTP requests",
    ["method", "endpoint", "status"]
)

# Measures HTTP request processing time in seconds.
HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request processing duration in seconds",
    ["method", "endpoint"]
)

# Counts successful and failed prediction operations.
PREDICTIONS = Counter(
    "predictions",
    "Total number of successful and failed predictions",
    ["outcome", "result"]
)

# Shows whether the ML model loaded successfully.
MODEL_LOADED = Gauge(
    "model_loaded",
    "Whether the machine learning model is loaded (1=yes, 0=no)"
)

MODEL_LOADED.set(0)


# STEP 4: LOAD MACHINE LEARNING MODEL

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Please run train_model.py first."
        )

    return joblib.load(MODEL_PATH)


try:
    model = load_model()

    MODEL_LOADED.set(1)

    logger.info(
        "event=model_loaded model_file=%s",
        MODEL_PATH.name
    )

except Exception:
    MODEL_LOADED.set(0)

    logger.exception(
        "event=model_load_failed model_file=%s",
        MODEL_PATH.name
    )
    raise


# STEP 5: REQUEST LOGGING AND HTTP METRICS

@app.before_request
def before_request_logging():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


@app.after_request
def after_request_logging(response):

    duration_seconds = (
        time.perf_counter() - g.start_time
    )

    duration_ms = duration_seconds * 1000

    request_id = getattr(
        g,
        "request_id",
        "unknown"
    )

    response.headers["X-Request-ID"] = request_id

    # Use the Flask route rule to avoid recording arbitrary URL values.
    endpoint = (
        request.url_rule.rule
        if request.url_rule is not None
        else "unmatched"
    )

    # Record HTTP request count and response time.
    HTTP_REQUESTS.labels(
        method=request.method,
        endpoint=endpoint,
        status=str(response.status_code)
    ).inc()

    HTTP_REQUEST_DURATION.labels(
        method=request.method,
        endpoint=endpoint
    ).observe(duration_seconds)

    level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        level,
        "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
        request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


# STEP 6: HOME ENDPOINT

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Result Prediction API is running",
        "status": "ok"
    }), 200


# STEP 7: HEALTH ENDPOINT

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction"
    }), 200


# STEP 8: PROMETHEUS METRICS ENDPOINT

@app.route("/metrics", methods=["GET"])
def metrics():
    return Response(
        generate_latest(),
        mimetype=CONTENT_TYPE_LATEST
    )


# STEP 9: PREDICTION ENDPOINT

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    # Validate JSON request
    if not isinstance(data, dict):
        PREDICTIONS.labels(
            outcome="failure",
            result="none"
        ).inc()

        logger.warning(
            "event=validation_error reason=invalid_json"
        )

        return jsonify({
            "error": "Request body must be a valid JSON object"
        }), 400

    # Check required fields
    missing_fields = [
        field for field in FEATURES
        if field not in data
    ]

    if missing_fields:
        PREDICTIONS.labels(
            outcome="failure",
            result="none"
        ).inc()

        logger.warning(
            "event=validation_error missing_fields=%s",
            missing_fields
        )

        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    # Validate numeric values
    input_data = {}

    for field in FEATURES:
        value = data[field]

        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not 0 <= value <= 100
        ):
            PREDICTIONS.labels(
                outcome="failure",
                result="none"
            ).inc()

            logger.warning(
                "event=validation_error field=%s reason=invalid_numeric_value",
                field
            )

            return jsonify({
                "error": "All input fields must be numeric values between 0 and 100",
                "invalid_field": field
            }), 400

        input_data[field] = float(value)

    try:
        logger.info(
            "event=prediction_request feature_names=%s",
            FEATURES
        )

        input_df = pd.DataFrame(
            [input_data],
            columns=FEATURES
        )

        prediction = model.predict(input_df)[0]

        # Convert model output to PASS or FAIL
        if str(prediction).strip().upper() in ["1", "1.0", "PASS"]:
            result = "PASS"
            prediction_code = 1

        elif str(prediction).strip().upper() in ["0", "0.0", "FAIL"]:
            result = "FAIL"
            prediction_code = 0

        else:
            raise ValueError(
                "Unexpected prediction output from model"
            )

        # Record successful prediction.
        PREDICTIONS.labels(
            outcome="success",
            result=result
        ).inc()

        logger.info(
            "event=prediction_success prediction=%s",
            result
        )

        return jsonify({
            "prediction": result,
            "prediction_code": prediction_code
        }), 200

    except Exception:
        # Record failed prediction.
        PREDICTIONS.labels(
            outcome="failure",
            result="none"
        ).inc()

        logger.exception(
            "event=prediction_failed request_id=%s",
            getattr(g, "request_id", "unknown")
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# STEP 10: HANDLE UNKNOWN ROUTES

@app.errorhandler(404)
def not_found(error):
    logger.warning(
        "event=route_not_found path=%s",
        request.path
    )

    return jsonify({
        "error": "Route not found"
    }), 404


# STEP 11: START APPLICATION

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))

    logger.info(
        "event=application_starting port=%s",
        port
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
