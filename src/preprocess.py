from pathlib import Path
import sys
from typing import cast

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Target cleaning
# ---------------------------------------------------------------------------


def clean_target_is_fraud(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return validated target and row-validity mask for is_fraud.

    Invalid or missing labels are flagged so caller can drop those rows
    explicitly instead of silently coercing labels.
    """
    if "is_fraud" not in df.columns:
        raise KeyError("Expected column 'is_fraud' not found in dataset.")

    target = cast(pd.Series, pd.to_numeric(df["is_fraud"], errors="coerce"))
    valid_mask = cast(pd.Series, target.isin([0, 1]))
    target = target.where(valid_mask)

    return target, valid_mask


# ---------------------------------------------------------------------------
# Geographic feature engineering
# ---------------------------------------------------------------------------


def haversine_km(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance between two geographic coordinates.

    Parameters
    ----------
    lat1, lon1 : array-like
        Customer latitude and longitude.
    lat2, lon2 : array-like
        Merchant latitude and longitude.

    Returns
    -------
    array-like
        Distance in kilometres.
    """
    radius_km = 6371.0

    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    )

    return 2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------


def preprocess_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean raw transaction data and create model-ready features.

    This function performs data cleaning and feature engineering only.
    Model-dependent preprocessing such as imputation and categorical
    encoding is intentionally handled in train_model.py.
    """

    df = df.copy()

    # 1) Remove exact duplicate records.
    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before_rows - len(df)

    # 2) Clean target and drop rows with invalid labels.
    target, valid_target_mask = clean_target_is_fraud(df)
    invalid_target_rows = int((~valid_target_mask).sum())

    if invalid_target_rows > 0:
        print(
            f"Warning: {invalid_target_rows} rows dropped due to invalid "
            "or missing is_fraud labels."
        )
        valid_idx = valid_target_mask.to_numpy(dtype=bool)
        df = df.loc[valid_idx].reset_index(drop=True)
        target = target.loc[valid_idx].reset_index(drop=True)

    y = target.astype(int)

    # 3) Parse transaction datetime and extract temporal features.
    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"],
            dayfirst=True,
            errors="coerce",
        )

        invalid_transaction_dates = int(df["trans_date_trans_time"].isna().sum())
        if invalid_transaction_dates > 0:
            print(
                "Warning: "
                f"{invalid_transaction_dates} invalid transaction dates "
                "converted to NaT."
            )

        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
        df["trans_month"] = df["trans_date_trans_time"].dt.month

    # 4) Parse dob and compute customer age.
    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")

        invalid_dob = int(df["dob"].isna().sum())
        if invalid_dob > 0:
            print(f"Warning: {invalid_dob} invalid DOB values " "converted to NaT.")

        if "trans_date_trans_time" in df.columns:
            customer_age = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
            df["customer_age"] = customer_age.clip(lower=0, upper=110)

    # 5) Merchant distance feature.
    geo_cols = ["lat", "long", "merch_lat", "merch_long"]
    if all(col in df.columns for col in geo_cols):
        df["merchant_distance_km"] = haversine_km(
            df["lat"],
            df["long"],
            df["merch_lat"],
            df["merch_long"],
        )

        distance = df["merchant_distance_km"]
        print("Merchant distance statistics:")
        print(f"  Minimum : {distance.min():.2f} km")
        print(f"  Median  : {distance.median():.2f} km")
        print(f"  Maximum : {distance.max():.2f} km")

    # 6) Remove raw target/id/date columns replaced by engineered features.
    drop_cols = ["is_fraud", "trans_num", "dob", "trans_date_trans_time"]
    drop_cols = [col for col in drop_cols if col in df.columns]
    x = df.drop(columns=drop_cols)

    # 7) Remove extremely high-cardinality categoricals for baseline model.
    high_cardinality_cols = [
        col
        for col in x.select_dtypes(include=["object", "category", "string"]).columns
        if x[col].nunique(dropna=True) > 200
    ]

    if high_cardinality_cols:
        print("High-cardinality categorical columns removed " "(>200 unique values):")
        for col in high_cardinality_cols:
            unique_count = x[col].nunique(dropna=True)
            print(f"  - {col}: {unique_count:,} unique values")
        x = x.drop(columns=high_cardinality_cols)
    else:
        print("No high-cardinality categorical columns required removal.")

    # 8) Missing-value report.
    missing = x.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nMissing-value report:")
    if missing.empty:
        print("  No missing values in features.")
    else:
        print(missing.to_string())

    # 9) Final summary.
    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Rows before deduplication  : {before_rows:,}")
    print(f"Rows after deduplication   : {len(df):,}")
    print(f"Duplicates removed         : {removed_duplicates:,}")
    print(f"Invalid-target rows dropped: {invalid_target_rows:,}")
    print(f"Features after preprocessing: {x.shape[1]:,}")
    print(f"Fraudulent transactions    : {y.sum():,}")
    print(f"Non-fraudulent transactions: {(y == 0).sum():,}")
    print(f"Fraud rate                 : {y.mean():.4%}")

    return x, y


# ---------------------------------------------------------------------------
# Save processed dataset
# ---------------------------------------------------------------------------


def save_cleaned_dataset(
    x: pd.DataFrame,
    y: pd.Series,
) -> None:
    """Save cleaned features and target as a CSV file."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_df = x.copy()
    output_df["is_fraud"] = y.to_numpy()
    output_df.to_csv(PROCESSED_FILE, index=False)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------


def run_preprocessing() -> None:
    """Load raw data, preprocess it, and save the cleaned dataset."""

    print("Loading raw dataset...")
    df = load_raw_data()
    print(f"Raw dataset loaded: {len(df):,} rows, {df.shape[1]} columns.")

    x, y = preprocess_dataframe(df)
    save_cleaned_dataset(x, y)

    print("\nDataset summary")
    print("---------------------")
    print(f"Rows                     : {len(x):,}")
    print(f"Features after processing: {x.shape[1]:,}")
    print(f"Fraud rate               : {y.mean():.4%}")
    print("\nCleaned dataset saved to:")
    print(PROCESSED_FILE)


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_preprocessing()
