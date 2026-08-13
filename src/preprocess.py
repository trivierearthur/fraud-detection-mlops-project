from pathlib import Path
import sys

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    try:
        from fraud_detection_mlops_project.src.data_loader import load_raw_data
    except ModuleNotFoundError:
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from data_loader import load_raw_data

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "fraud_data_cleaned.csv"


def clean_target_is_fraud(df: pd.DataFrame) -> pd.Series:
    """Return a clean binary target vector from the is_fraud column."""
    if "is_fraud" not in df.columns:
        raise KeyError("Expected column 'is_fraud' not found in dataset")

    y = (df["is_fraud"] == 1) | (df["is_fraud"] == "1")
    y = y.astype(int)
    return y


def haversine_km(lat1, lon1, lat2, lon2):
    """Compute great-circle distance between card and merchant points in km."""
    radius_km = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )
    return 2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def preprocess_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create cleaned features X and target y from raw transactions."""
    df = df.copy()

    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before_rows - len(df)

    y = clean_target_is_fraud(df)

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"], dayfirst=True, errors="coerce"
        )
        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
        df["trans_month"] = df["trans_date_trans_time"].dt.month

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")
        if "trans_date_trans_time" in df.columns:
            customer_age = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
            df["customer_age"] = customer_age.clip(lower=0, upper=110)

    geo_cols = ["lat", "long", "merch_lat", "merch_long"]
    if all(col in df.columns for col in geo_cols):
        df["merchant_distance_km"] = haversine_km(
            df["lat"], df["long"], df["merch_lat"], df["merch_long"]
        )

    drop_cols = ["is_fraud", "trans_num", "dob", "trans_date_trans_time"]
    drop_cols = [col for col in drop_cols if col in df.columns]
    x = df.drop(columns=drop_cols)

    # Drop very high-cardinality string columns to keep one-hot
    # encoding stable.
    high_cardinality_cols = [
        col
        for col in x.select_dtypes(include=["object"]).columns
        if x[col].nunique(dropna=True) > 200
    ]
    if high_cardinality_cols:
        x = x.drop(columns=high_cardinality_cols)

    print("Preprocessing summary")
    print(f"Rows before deduplication: {before_rows}")
    print(f"Rows after deduplication: {len(df)}")
    print(f"Duplicates removed: {removed_duplicates}")

    missing = x.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values in features.")
    else:
        print("Missing values by feature:")
        print(missing.to_string())

    return x, y


def save_cleaned_dataset(x: pd.DataFrame, y: pd.Series) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_df = x.copy()
    output_df["is_fraud"] = y.values
    output_df.to_csv(PROCESSED_FILE, index=False)


def run_preprocessing() -> None:
    df = load_raw_data()
    x, y = preprocess_dataframe(df)

    save_cleaned_dataset(x, y)

    print("Dataset summary")
    print(f"Rows: {len(x)}")
    print(f"Features after preprocessing: {x.shape[1]}")
    print(f"Fraud rate: {y.mean():.4f}")
    print(f"\nCleaned dataset saved to: {PROCESSED_FILE}")


if __name__ == "__main__":
    run_preprocessing()
