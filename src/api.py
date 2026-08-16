from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_FILE = PROJECT_ROOT / "models" / "fraud_model.joblib"


app = Flask(__name__)


# Load the trained pipeline when the API starts
if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Saved model not found at {MODEL_FILE}. " "Run train_model.py first."
    )

model = joblib.load(MODEL_FILE)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Request must contain JSON data"}), 400

    try:
        transaction = pd.DataFrame([data])

        fraud_probability = float(model.predict_proba(transaction)[0, 1])

        fraud_prediction = int(model.predict(transaction)[0])

        return jsonify(
            {
                "fraud_probability": fraud_probability,
                "fraud_prediction": fraud_prediction,
            }
        )

    except Exception as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
