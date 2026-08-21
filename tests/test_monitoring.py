"""Automated tests for drift monitoring (src/monitoring.py)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.monitoring import monitor_month


def _make_reference_data(n=200, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "amt": rng.normal(50, 15, n),
            "trans_hour": rng.integers(0, 24, n),
            "trans_dayofweek": rng.integers(0, 7, n),
            "customer_age": rng.normal(40, 10, n),
            "merchant_distance_km": rng.normal(10, 5, n),
            "city_pop": rng.integers(1000, 500000, n),
            "category": rng.choice(["grocery_pos", "gas_transport", "shopping_net"], n),
            "state": rng.choice(["NY", "PA", "MA"], n),
        }
    )


def test_unchanged_data_does_not_trigger_drift():
    """Comparing a dataset against itself should never report drift."""
    reference = _make_reference_data()
    current = reference.copy()

    result = monitor_month(reference, current, month=1)

    assert result["drift_detected"] is False
    assert result["drifted_features"] == []


def test_shifted_data_triggers_drift():
    """A deliberately shifted distribution should be flagged as drift."""
    reference = _make_reference_data()

    current = reference.copy()
    # Shift numeric features far outside the reference distribution.
    current["amt"] = current["amt"] + 200
    current["customer_age"] = current["customer_age"] + 40
    current["merchant_distance_km"] = current["merchant_distance_km"] + 100
    # Flip the categorical distribution entirely.
    current["category"] = "shopping_net"
    current["state"] = "MA"

    result = monitor_month(reference, current, month=2)

    assert result["drift_detected"] is True
    assert len(result["drifted_features"]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
