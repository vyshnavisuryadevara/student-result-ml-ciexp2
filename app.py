
from flask import Flask, jsonify, request

# Initialize Flask application
app = Flask(__name__)

# Home route
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Welcome to Student Result Prediction API",
        "service": "student-result-prediction"
    }), 200


# Health check route
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# API information route
@app.route("/api", methods=["GET"])
def api_info():
    return jsonify({
        "project": "Student Result Prediction",
        "framework": "Flask",
        "message": "API is running successfully"
    }), 200


# Run Flask application
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )
