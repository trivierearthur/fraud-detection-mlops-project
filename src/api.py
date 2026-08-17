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


# Raw features required by the prediction pipeline.
# is_fraud is intentionally excluded because it is the target.
REQUIRED_FEATURES = [
    "trans_date_trans_time",
    "merchant",
    "category",
    "amt",
    "city",
    "state",
    "lat",
    "long",
    "city_pop",
    "job",
    "dob",
    "trans_num",
    "merch_lat",
    "merch_long",
]


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    # Reject empty or missing JSON requests
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    # Check that all required raw features are present
    missing_features = [feature for feature in REQUIRED_FEATURES if feature not in data]

    if missing_features:
        return (
            jsonify(
                {
                    "error": "Missing required features",
                    "missing": missing_features,
                }
            ),
            400,
        )

    try:
        # Convert the incoming transaction to a DataFrame
        transaction = pd.DataFrame([data])

        # Generate fraud probability
        fraud_probability = float(model.predict_proba(transaction)[0, 1])

        # Generate binary fraud prediction
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
