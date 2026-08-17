from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

MODELS_DIR = PROJECT_ROOT / "models"
CURRENT_MODEL_FILE = MODELS_DIR / "current_model.txt"

app = Flask(__name__)


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
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "Request must contain JSON data"}), 400

    if not data:
        return jsonify({"error": "Request body cannot be empty"}), 400

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
