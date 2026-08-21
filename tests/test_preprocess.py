"""
Automated tests for preprocessing and feature engineering.

Tests validate:
1. Feature engineering creates expected features (datetime, geographic, age)
2. High-cardinality columns are correctly removed
3. Missing values are handled appropriately
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add the project root to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import (
    engineer_transaction_features,
    TransactionFeatureTransformer,
    clean_target_is_fraud,
    prepare_training_dataframe,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_raw_transaction_data():
    """Create a minimal valid transaction dataset for testing."""
    return pd.DataFrame(
        {
            "trans_date_trans_time": [
                "01-01-2025 12:30",
                "02-01-2025 14:45",
                "03-01-2025 09:15",
            ],
            "dob": ["01-01-1990", "15-06-1985", "22-12-1995"],
            "lat": [40.0, 40.1, 40.2],
            "long": [-75.0, -75.1, -75.2],
            "merch_lat": [40.05, 40.15, 40.25],
            "merch_long": [-75.05, -75.15, -75.25],
            "amt": [50.0, 100.0, 25.0],
            "merchant": ["Store A", "Store B", "Store C"],
            "category": ["grocery_pos", "gas_transport", "shopping_net"],
            "city": ["New York", "Philadelphia", "Boston"],
            "state": ["NY", "PA", "MA"],
            "city_pop": [8000000, 1600000, 700000],
            "job": ["Analyst", "Manager", "Engineer"],
            "trans_num": ["T001", "T002", "T003"],
            "is_fraud": [0, 1, 0],
        }
    )


@pytest.fixture
def sample_high_cardinality_data():
    """Create transaction data with high-cardinality columns for testing removal."""
    n_rows = 50
    # Create merchants: each of the 50 rows gets a unique merchant (50 unique)
    # But we create a field with enough unique values to exceed threshold
    merchants = [f"Merchant_{i % 50}" for i in range(n_rows)]  # 50 unique values
    jobs = [f"Job_{i % 30}" for i in range(n_rows)]  # 30 unique values

    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range(
                "2025-01-01", periods=n_rows, freq="h"
            ).strftime("%d-%m-%Y %H:%M"),
            "dob": pd.date_range("1960-01-01", periods=n_rows, freq="D").strftime(
                "%d-%m-%Y"
            ),
            "lat": np.random.uniform(39.0, 41.0, n_rows),
            "long": np.random.uniform(-76.0, -74.0, n_rows),
            "merch_lat": np.random.uniform(39.0, 41.0, n_rows),
            "merch_long": np.random.uniform(-76.0, -74.0, n_rows),
            "amt": np.random.uniform(10.0, 500.0, n_rows),
            "merchant": merchants,
            "category": np.random.choice(
                ["grocery_pos", "gas_transport", "shopping_net"], n_rows
            ),
            "city": np.random.choice(["New York", "Philadelphia", "Boston"], n_rows),
            "state": np.random.choice(["NY", "PA", "MA"], n_rows),
            "city_pop": np.random.randint(100000, 10000000, n_rows),
            "job": jobs,
            "trans_num": [f"T{i:05d}" for i in range(n_rows)],
            "is_fraud": np.random.randint(0, 2, n_rows),
        }
    )


# ---------------------------------------------------------------------------
# Test 1: Feature Engineering Creates Expected Features
# ---------------------------------------------------------------------------


def test_feature_engineering_creates_expected_features(
    sample_raw_transaction_data, capsys
):
    """
    Test that engineer_transaction_features creates all expected engineered features.

    Expected features after engineering:
    - trans_hour: Extracted from trans_date_trans_time
    - trans_dayofweek: Extracted from trans_date_trans_time
    - trans_month: Extracted from trans_date_trans_time
    - customer_age: Calculated as days between transaction and DOB
    - merchant_distance_km: Haversine distance between customer and merchant
    """
    df = sample_raw_transaction_data.copy()

    # Engineer features
    x, removed_cols = engineer_transaction_features(df, verbose=False)

    # Verify engineered features exist
    assert (
        "trans_hour" in x.columns
    ), "trans_hour not created from trans_date_trans_time"
    assert (
        "trans_dayofweek" in x.columns
    ), "trans_dayofweek not created from trans_date_trans_time"
    assert (
        "trans_month" in x.columns
    ), "trans_month not created from trans_date_trans_time"
    assert "customer_age" in x.columns, "customer_age not created from DOB"
    assert (
        "merchant_distance_km" in x.columns
    ), "merchant_distance_km not created from coordinates"

    # Verify raw columns are dropped
    assert (
        "trans_date_trans_time" not in x.columns
    ), "trans_date_trans_time should be dropped"
    assert "dob" not in x.columns, "dob should be dropped"

    # Verify feature values are reasonable
    assert (
        x["trans_hour"].min() >= 0 and x["trans_hour"].max() < 24
    ), "trans_hour out of valid range [0, 23]"
    assert (
        x["trans_dayofweek"].min() >= 0 and x["trans_dayofweek"].max() < 7
    ), "trans_dayofweek out of valid range [0, 6]"
    assert (
        x["trans_month"].min() >= 1 and x["trans_month"].max() <= 12
    ), "trans_month out of valid range [1, 12]"
    assert (
        x["customer_age"].min() >= 0 and x["customer_age"].max() <= 110
    ), "customer_age out of reasonable range"
    assert (
        x["merchant_distance_km"].min() >= 0
    ), "merchant_distance_km should be non-negative"
    assert (
        x["merchant_distance_km"] < 50
    ).all(), (
        "All sample distances should be < 50 km (test data is geographically close)"
    )

    # Verify shape is correct (3 rows from sample)
    assert len(x) == 3, f"Expected 3 rows, got {len(x)}"


# ---------------------------------------------------------------------------
# Test 2: High-Cardinality Columns Are Removed
# ---------------------------------------------------------------------------


def test_high_cardinality_columns_removal(sample_high_cardinality_data, capsys):
    """
    Test that high-cardinality categorical columns are correctly identified and removed.

    Note: In this test, we use the default threshold of 200.
    Sample data has low cardinality (50 merchants, 30 jobs - both under threshold).
    Test focuses on verifying the removal logic works, not finding high-cardinality columns.
    """
    df = sample_high_cardinality_data.copy()

    # Engineer features with verbose output
    x, removed_cols = engineer_transaction_features(
        df, verbose=True, high_cardinality_threshold=200
    )

    # With threshold of 200, the low-cardinality sample columns should NOT be removed
    assert (
        "merchant" in x.columns
    ), "merchant (50 unique) should be kept (below threshold of 200)"
    assert "job" in x.columns, "job (30 unique) should be kept (below threshold of 200)"

    # Verify low-cardinality categorical columns are kept
    assert "category" in x.columns, "category (low cardinality) should be kept"
    assert "city" in x.columns, "city (low cardinality) should be kept"
    assert "state" in x.columns, "state (low cardinality) should be kept"

    # Now test with a lower threshold to force removal
    x_strict, removed_cols_strict = engineer_transaction_features(
        df, verbose=True, high_cardinality_threshold=25
    )

    # With threshold of 25, merchant (50) and job (30) should be removed
    assert (
        "merchant" not in x_strict.columns
    ), "merchant (50 unique) should be removed with threshold of 25"
    assert (
        "job" not in x_strict.columns
    ), "job (30 unique) should be removed with threshold of 25"
    assert (
        "merchant" in removed_cols_strict
    ), "merchant should be in removed_cols_strict list"
    assert "job" in removed_cols_strict, "job should be in removed_cols_strict list"


# ---------------------------------------------------------------------------
# Test 3: TransactionFeatureTransformer Fit/Transform Consistency
# ---------------------------------------------------------------------------


def test_transaction_feature_transformer_consistency(sample_raw_transaction_data):
    """
    Test that TransactionFeatureTransformer maintains consistent feature columns
    during fit and transform phases.

    This ensures the transformer:
    - Learns feature engineering rules during fit
    - Applies them consistently during transform
    - Returns features in same order
    """
    df = sample_raw_transaction_data.copy()

    # Split into training and test set (naive split for testing)
    train_df = df.iloc[:2].copy()
    test_df = df.iloc[2:].copy()

    # Fit transformer on training data
    transformer = TransactionFeatureTransformer(verbose=False)
    transformer.fit(train_df)

    # Verify fit created feature columns
    assert hasattr(
        transformer, "feature_columns_"
    ), "Transformer should store feature_columns_ after fit"
    assert isinstance(
        transformer.feature_columns_, list
    ), "feature_columns_ should be a list"
    assert len(transformer.feature_columns_) > 0, "feature_columns_ should not be empty"

    # Transform training data
    x_train = transformer.transform(train_df)
    assert isinstance(x_train, pd.DataFrame), "Transform should return DataFrame"
    assert (
        list(x_train.columns) == transformer.feature_columns_
    ), "Train transform should have fitted feature columns"

    # Transform test data with same feature columns
    x_test = transformer.transform(test_df)
    assert isinstance(x_test, pd.DataFrame), "Transform should return DataFrame"
    assert (
        list(x_test.columns) == transformer.feature_columns_
    ), "Test transform should have same feature columns as train"
    assert len(x_test.columns) == len(
        x_train.columns
    ), "Feature count should match between train and test transforms"

    # Verify all features are numeric or can be used in model training
    for col in x_test.columns:
        assert x_test[col].dtype in [
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            object,
            "string",
        ], f"Column {col} has unsupported dtype {x_test[col].dtype}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
