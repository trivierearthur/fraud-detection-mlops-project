from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from src.auth import require_api_key

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_MODEL_FILE = MODELS_DIR / "current_model.txt"

app = Flask(__name__)


# Expected input types for a raw transaction
EXPECTED_TYPES = {
    "trans_date_trans_time": str,
    "merchant": str,
    "category": str,
    "amt": (int, float),
    "city": str,
    "state": str,
    "lat": (int, float),
    "long": (int, float),
    "city_pop": (int, float),
    "job": str,
    "dob": str,
    "trans_num": str,
    "merch_lat": (int, float),
    "merch_long": (int, float),
}


def load_current_model():
    """Load the model specified by current_model.txt."""

    if not CURRENT_MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Current model file not found at {CURRENT_MODEL_FILE}. "
            "Run train_model.py first."
        )

    model_filename = CURRENT_MODEL_FILE.read_text(encoding="utf-8").strip()

    if not model_filename:
        raise ValueError(f"{CURRENT_MODEL_FILE} is empty.")

    model_path = MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model specified in current_model.txt was not found: " f"{model_path}"
        )

    print(f"Loading model: {model_path}")

    return joblib.load(model_path)


# Load the current production model when the API starts
model = load_current_model()


@app.route("/health", methods=["GET"])
def health():
    """Check whether the API is running."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
@require_api_key
def predict():
    """Predict the fraud probability for one transaction."""

    data = request.get_json(silent=True)

    # Check that JSON was provided
    if data is None:
        return jsonify({"error": "Request must contain JSON data"}), 400

    # Check that the JSON object is not empty
    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

    # Check for missing required fields
    missing_fields = [field for field in EXPECTED_TYPES if field not in data]

    if missing_fields:
        return (
            jsonify(
                {
                    "error": "Missing required fields",
                    "fields": missing_fields,
                }
            ),
            400,
        )

    # Check data types
    for field, expected_type in EXPECTED_TYPES.items():
        if not isinstance(data[field], expected_type):
            expected = (
                "number" if isinstance(expected_type, tuple) else expected_type.__name__
            )

            return (
                jsonify(
                    {
                        "error": f"Invalid type for '{field}'",
                        "expected": expected,
                        "received": type(data[field]).__name__,
                    }
                ),
                400,
            )

    # Validate transaction datetime format
    try:
        datetime.strptime(
            data["trans_date_trans_time"],
            "%d-%m-%Y %H:%M",
        )
    except ValueError:
        return (
            jsonify(
                {
                    "error": (
                        "Invalid trans_date_trans_time format. "
                        "Expected DD-MM-YYYY HH:MM"
                    )
                }
            ),
            400,
        )

    try:
        # Convert the validated transaction into a DataFrame
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

    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    except Exception:
        return jsonify({"error": "Internal prediction error"}), 500


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
    )
