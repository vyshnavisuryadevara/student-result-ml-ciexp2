
import json
import sys

MINIMUM_ACCURACY = 0.99

print("Reading model evaluation metrics...")

try:
    with open("metrics.json", "r") as file:
        metrics = json.load(file)

    accuracy = float(metrics["accuracy"])

except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
    print("QUALITY GATE FAILED")
    print(f"Could not read valid model metrics: {error}")
    sys.exit(1)

print("Model Accuracy :", round(accuracy, 4))
print("Required Accuracy:", MINIMUM_ACCURACY)

if not 0.0 <= accuracy <= 1.0:
    print("QUALITY GATE FAILED")
    print("Accuracy must be between 0 and 1.")
    sys.exit(1)

if accuracy < MINIMUM_ACCURACY:
    print("QUALITY GATE FAILED")
    print("Model performance is below the required threshold.")
    sys.exit(1)

print("QUALITY GATE PASSED")
print("Model performance satisfies the required threshold.")
sys.exit(0)
