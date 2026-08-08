from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_FILE = RAW_DATA_DIR / "fraud_data.csv"
SOURCE_DATA_FILE = Path(r"C:\Users\trivi\Desktop\from model to prod\fraud_data.csv")


def load_raw_data():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SOURCE_DATA_FILE.exists():
        df = pd.read_csv(SOURCE_DATA_FILE)
        df.to_csv(RAW_DATA_FILE, index=False)
        return df

    if RAW_DATA_FILE.exists():
        return pd.read_csv(RAW_DATA_FILE)

    raise FileNotFoundError(
        f"No raw data found at {SOURCE_DATA_FILE} or {RAW_DATA_FILE}"
    )
