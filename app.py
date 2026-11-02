
from flask import Flask, jsonify

# Create Flask application
app = Flask(__name__)


# Home page
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Student Result Prediction API",
        "status": "success",
        "service": "student-result-prediction"
    }), 200


# Health check
@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


# API status
@app.route("/api", methods=["GET"])
def api_status():
    return jsonify({
        "message": "API is working properly",
        "status": "running"
    }), 200


# Handle invalid URLs
@app.errorhandler(404)
def page_not_found(error):
    return jsonify({
        "status": "error",
        "message": "Endpoint not found"
    }), 404


# Start application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
