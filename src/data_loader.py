from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "fraud_data.csv"


def load_raw_data() -> pd.DataFrame:
    """Load the project's raw fraud dataset."""

    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Raw data not found at {RAW_DATA_FILE}. "
            "Make sure data/raw/fraud_data.csv is available."
        )

    return pd.read_csv(RAW_DATA_FILE)
