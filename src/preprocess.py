import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# ---------------------------------------------------------------------------
# Project paths and imports
# ---------------------------------------------------------------------------

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    try:
        from fraud_detection_mlops_project.src.data_loader import (  # pyright: ignore[reportMissingImports]
            load_raw_data,
        )
    except ModuleNotFoundError:
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from data_loader import load_raw_data


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_FILE = PROCESSED_DIR / "fraud_data_cleaned.csv"
TARGET_COLUMN = "is_fraud"
RAW_FEATURE_DROP_COLUMNS = [
    "is_fraud",
    "trans_num",
    "dob",
    "trans_date_trans_time",
]
HIGH_CARDINALITY_THRESHOLD = 200


# ---------------------------------------------------------------------------
# Target cleaning
# ---------------------------------------------------------------------------


def clean_target_is_fraud(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Return validated target and row-validity mask for is_fraud.

    Invalid or missing labels are flagged so caller can drop those rows
    explicitly instead of silently coercing labels.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError("Expected column 'is_fraud' not found in dataset.")

    target = cast(pd.Series, pd.to_numeric(df[TARGET_COLUMN], errors="coerce"))
    valid_mask = cast(pd.Series, target.isin([0, 1]))
    target = target.where(valid_mask)

    return target, valid_mask


# ---------------------------------------------------------------------------
# Geographic feature engineering
# ---------------------------------------------------------------------------


def haversine_km(
    lat1: pd.Series,
    lon1: pd.Series,
    lat2: pd.Series,
    lon2: pd.Series,
) -> Any:
    """
    Compute great-circle distance between two geographic coordinates.

    Returns a Series at runtime (numpy ufuncs preserve pandas types via
    __array_ufunc__), but numpy's stubs type elementwise ops as ndarray,
    so the static return type is left as Any rather than fighting the stub.

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

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2

    return 2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def prepare_training_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, dict[str, int]]:
    """Drop unusable training rows and return raw records plus target."""

    df = df.copy()
    before_rows = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed_duplicates = before_rows - len(df)

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
    summary = {
        "before_rows": before_rows,
        "after_rows": len(df),
        "removed_duplicates": removed_duplicates,
        "invalid_target_rows": invalid_target_rows,
    }
    return df, y, summary


def _print_missing_value_report(x: pd.DataFrame) -> None:
    missing = x.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\nMissing-value report:")
    if missing.empty:
        print("  No missing values in features.")
    else:
        print(missing.to_string())


def engineer_transaction_features(
    df: pd.DataFrame,
    high_cardinality_cols: list[str] | None = None,
    high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """Create model features from raw transactions."""

    df = df.copy()

    if "trans_date_trans_time" in df.columns:
        df["trans_date_trans_time"] = pd.to_datetime(
            df["trans_date_trans_time"],
            dayfirst=True,
            errors="coerce",
        )

        invalid_transaction_dates = int(df["trans_date_trans_time"].isna().sum())
        if verbose and invalid_transaction_dates > 0:
            print(
                "Warning: "
                f"{invalid_transaction_dates} invalid transaction dates "
                "converted to NaT."
            )

        df["trans_hour"] = df["trans_date_trans_time"].dt.hour
        df["trans_dayofweek"] = df["trans_date_trans_time"].dt.dayofweek
        df["trans_month"] = df["trans_date_trans_time"].dt.month

    if "dob" in df.columns:
        df["dob"] = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")

        invalid_dob = int(df["dob"].isna().sum())
        if verbose and invalid_dob > 0:
            print(f"Warning: {invalid_dob} invalid DOB values converted to NaT.")

        if "trans_date_trans_time" in df.columns:
            customer_age = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
            df["customer_age"] = customer_age.clip(lower=0, upper=110)

    geo_cols = ["lat", "long", "merch_lat", "merch_long"]
    if all(col in df.columns for col in geo_cols):
        df["merchant_distance_km"] = haversine_km(
            df["lat"],
            df["long"],
            df["merch_lat"],
            df["merch_long"],
        )

        if verbose:
            distance = df["merchant_distance_km"]
            print("Merchant distance statistics:")
            print(f"  Minimum : {distance.min():.2f} km")
            print(f"  Median  : {distance.median():.2f} km")
            print(f"  Maximum : {distance.max():.2f} km")

    drop_cols = [col for col in RAW_FEATURE_DROP_COLUMNS if col in df.columns]
    x = df.drop(columns=drop_cols)

    if high_cardinality_cols is None:
        high_cardinality_cols = [
            col
            for col in x.select_dtypes(include=["object", "category", "string"]).columns
            if x[col].nunique(dropna=True) > high_cardinality_threshold
        ]

    if high_cardinality_cols:
        if verbose:
            print(
                "High-cardinality categorical columns removed "
                f"(>{high_cardinality_threshold} unique values):"
            )
            for col in high_cardinality_cols:
                unique_count = x[col].nunique(dropna=True)
                print(f"  - {col}: {unique_count:,} unique values")
        x = x.drop(columns=[col for col in high_cardinality_cols if col in x.columns])
    elif verbose:
        print("No high-cardinality categorical columns required removal.")

    if verbose:
        _print_missing_value_report(x)

    return x, list(high_cardinality_cols)


class TransactionFeatureTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible raw transaction feature engineering."""

    def __init__(
        self,
        high_cardinality_threshold: int = HIGH_CARDINALITY_THRESHOLD,
        verbose: bool = False,
    ) -> None:
        self.high_cardinality_threshold = high_cardinality_threshold
        self.verbose = verbose

    def fit(
        self,
        x: pd.DataFrame,
        y: Any = None,
    ) -> "TransactionFeatureTransformer":
        engineered_df, high_cardinality_cols = engineer_transaction_features(
            pd.DataFrame(x),
            high_cardinality_threshold=self.high_cardinality_threshold,
            verbose=self.verbose,
        )
        self.high_cardinality_cols_ = high_cardinality_cols
        self.feature_columns_ = engineered_df.columns.tolist()
        return self

    def transform(self, x: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self, "feature_columns_"):
            raise ValueError("TransactionFeatureTransformer must be fitted first.")

        engineered_df, _ = engineer_transaction_features(
            pd.DataFrame(x),
            high_cardinality_cols=self.high_cardinality_cols_,
            high_cardinality_threshold=self.high_cardinality_threshold,
            verbose=self.verbose,
        )

        for column in self.feature_columns_:
            if column not in engineered_df.columns:
                engineered_df[column] = pd.NA

        return engineered_df[self.feature_columns_].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main preprocessing
# ---------------------------------------------------------------------------


def preprocess_dataframe(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Clean raw transaction data and create model-ready features."""

    raw_df, y, summary = prepare_training_dataframe(df)
    transformer = TransactionFeatureTransformer(verbose=True)
    x = cast(pd.DataFrame, transformer.fit_transform(raw_df))

    print("\nPreprocessing summary")
    print("---------------------")
    print(f"Rows before deduplication  : {summary['before_rows']:,}")
    print(f"Rows after deduplication   : {summary['after_rows']:,}")
    print(f"Duplicates removed         : {summary['removed_duplicates']:,}")
    print(f"Invalid-target rows dropped: {summary['invalid_target_rows']:,}")
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
    output_df[TARGET_COLUMN] = y.to_numpy()
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
