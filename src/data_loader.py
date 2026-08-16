from pathlib import Path
import pandas as pd

# Resolve project-relative data paths once so callers can stay simple.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "fraud_data.csv"
SOURCE_DATA_FILE = Path(r"C:\Users\trivi\Desktop\from model to prod\fraud_data.csv")


def load_raw_data():
    """Load raw fraud data from source path or local project cache."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Preferred path: shared source file outside the repo.
    if SOURCE_DATA_FILE.exists():
        df = pd.read_csv(SOURCE_DATA_FILE)
        # Keep an in-repo copy so future runs do not depend on external path.
        df.to_csv(RAW_DATA_FILE, index=False)
        return df

    # Fallback path: previously cached dataset inside the project.
    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE)

    # Explicit error helps users know exactly where data is expected.
    raise FileNotFoundError(
        f"No raw data found at {SOURCE_DATA_FILE} or {RAW_DATA_FILE}"
    )
