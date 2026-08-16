from argparse import ArgumentParser
from pathlib import Path
import sys

import joblib
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
MODEL_FILE = PROJECT_ROOT / "models" / "fraud_model.joblib"

try:
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    try:
        from fraud_detection_mlops_project.src.data_loader import load_raw_data
    except ModuleNotFoundError:
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from data_loader import load_raw_data


def load_model():
    """Load the saved end-to-end fraud pipeline from disk."""
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Saved model not found at {MODEL_FILE}. Run train_model.py first."
        )
    return joblib.load(MODEL_FILE)


def load_prediction_input(
    input_csv: str | None,
    sample_row: int,
) -> pd.DataFrame:
    """Load raw transactions to score from CSV or from one sample row."""
    if input_csv is not None:
        input_path = Path(input_csv)
        if not input_path.exists():
            message = f"Prediction input file not found: {input_path}"
            raise FileNotFoundError(message)
        return pd.read_csv(input_path)

    # Fallback path for quick demos: score one raw row as if it just arrived.
    raw_df = load_raw_data()
    if sample_row < 0 or sample_row >= len(raw_df):
        raise IndexError(
            "sample_row="
            f"{sample_row} is outside the dataset range 0-{len(raw_df) - 1}."
        )

    print(
        "No input CSV provided. Using one raw transaction from the source "
        "dataset to simulate an incoming prediction request."
    )
    return raw_df.iloc[[sample_row]].copy()


def predict_transactions(model, transactions: pd.DataFrame) -> pd.DataFrame:
    """Return input rows enriched with fraud probability and class label."""
    fraud_probability = model.predict_proba(transactions)[:, 1]
    fraud_prediction = model.predict(transactions)

    results = transactions.copy()
    results["fraud_probability"] = fraud_probability
    results["fraud_prediction"] = fraud_prediction
    return results


def build_argument_parser() -> ArgumentParser:
    """Define command-line options for prediction input and output."""
    parser = ArgumentParser(
        description="Load the saved fraud pipeline and score new transactions."
    )
    parser.add_argument(
        "--input-csv",
        type=str,
        default=None,
        help="Path to a CSV file containing raw transaction records.",
    )
    parser.add_argument(
        "--sample-row",
        type=int,
        default=0,
        help=(
            "If --input-csv is omitted, use this row from the raw dataset to "
            "simulate a new incoming transaction."
        ),
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default=None,
        help="Optional path to save prediction results as CSV.",
    )
    return parser


def main() -> None:
    """CLI entrypoint for fraud scoring."""
    args = build_argument_parser().parse_args()

    # Load trained pipeline and score raw incoming transactions.
    model = load_model()
    transactions = load_prediction_input(args.input_csv, args.sample_row)
    predictions = predict_transactions(model, transactions)

    # Keep terminal output focused on key identifiers and model outputs.
    display_columns = [
        column
        for column in ["trans_num", "category", "amt", "is_fraud"]
        if column in predictions.columns
    ]
    display_columns.extend(["fraud_probability", "fraud_prediction"])

    print("Prediction results")
    print("------------------")
    print(predictions[display_columns].to_string(index=False))

    # Optional artifact for downstream analysis/reporting.
    if args.output_csv is not None:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(output_path, index=False)
        print(f"\nSaved predictions to: {output_path}")


if __name__ == "__main__":
    main()
