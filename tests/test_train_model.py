"""Automated tests for the training pipeline (src/train_model.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import prepare_training_dataframe
from src.train_model import build_pipeline


@pytest.fixture
def sample_training_data():
    """Small labelled transaction dataset with both fraud classes present."""
    n = 20
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "trans_date_trans_time": pd.date_range(
                "2025-01-01", periods=n, freq="h"
            ).strftime("%d-%m-%Y %H:%M"),
            "dob": pd.date_range("1970-01-01", periods=n, freq="365D").strftime(
                "%d-%m-%Y"
            ),
            "lat": rng.uniform(39.0, 41.0, n),
            "long": rng.uniform(-76.0, -74.0, n),
            "merch_lat": rng.uniform(39.0, 41.0, n),
            "merch_long": rng.uniform(-76.0, -74.0, n),
            "amt": rng.uniform(10.0, 500.0, n),
            "merchant": [f"Merchant_{i % 5}" for i in range(n)],
            "category": rng.choice(["grocery_pos", "gas_transport", "shopping_net"], n),
            "city": rng.choice(["New York", "Philadelphia", "Boston"], n),
            "state": rng.choice(["NY", "PA", "MA"], n),
            "city_pop": rng.integers(1000, 500000, n),
            "job": [f"Job_{i % 4}" for i in range(n)],
            "trans_num": [f"T{i:05d}" for i in range(n)],
            "is_fraud": [0, 1] * (n // 2),
        }
    )


def test_pipeline_can_train_and_predict(sample_training_data):
    """The full feature-engineering + preprocessing + model pipeline should fit and predict."""
    x, y, _ = prepare_training_dataframe(sample_training_data)

    pipeline = build_pipeline(x, LogisticRegression(max_iter=1000))
    pipeline.fit(x, y)

    predictions = pipeline.predict(x)

    assert len(predictions) == len(x)
    assert set(predictions).issubset({0, 1})


def test_probability_output_has_expected_shape(sample_training_data):
    """predict_proba should return one row per sample and two columns (class 0 / class 1)."""
    x, y, _ = prepare_training_dataframe(sample_training_data)

    pipeline = build_pipeline(x, LogisticRegression(max_iter=1000))
    pipeline.fit(x, y)

    probabilities = pipeline.predict_proba(x)

    assert probabilities.shape == (len(x), 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert ((probabilities >= 0) & (probabilities <= 1)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
