import sys
from pathlib import Path

import pytest

# Add the src directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from api import app
from auth import API_KEY


@pytest.fixture
def client():
    """Create a Flask test client."""
    app.config["TESTING"] = True

    with app.test_client() as client:
        yield client


VALID_TRANSACTION = {
    "trans_date_trans_time": "01-01-2025 12:30",
    "merchant": "Test Merchant",
    "category": "grocery_pos",
    "amt": 50.0,
    "city": "Test City",
    "state": "CA",
    "lat": 40.0,
    "long": -75.0,
    "city_pop": 100000,
    "job": "Analyst",
    "dob": "01-01-1990",
    "trans_num": "TEST123",
    "merch_lat": 40.1,
    "merch_long": -75.1,
}


def auth_headers():
    """Return authentication headers using the configured API key."""
    return {"Authorization": f"Bearer {API_KEY}"}


def test_health(client):
    """Health endpoint should return status ok."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.get_json() == {"status": "ok"}


def test_predict_without_api_key(client):
    """Prediction should require authentication."""
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
    )

    assert response.status_code == 401

    data = response.get_json()
    assert data["error"] == "Authentication required"


def test_predict_with_invalid_api_key(client):
    """Prediction should reject an invalid API key."""
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
        headers={"Authorization": "Bearer wrong-key"},
    )

    assert response.status_code == 401

    data = response.get_json()
    assert data["error"] == "Invalid API key"


def test_predict_with_valid_api_key(client):
    """Prediction should work with the configured API key."""
    response = client.post(
        "/predict",
        json=VALID_TRANSACTION,
        headers=auth_headers(),
    )

    assert response.status_code == 200

    data = response.get_json()

    assert "fraud_probability" in data
    assert "fraud_prediction" in data

    assert 0 <= data["fraud_probability"] <= 1
    assert data["fraud_prediction"] in [0, 1]


def test_predict_without_json(client):
    """Prediction should reject requests without JSON."""
    response = client.post(
        "/predict",
        headers=auth_headers(),
    )

    assert response.status_code == 400

    data = response.get_json()
    assert data["error"] == "Request must contain JSON data"


def test_predict_missing_fields(client):
    """Prediction should reject incomplete transactions."""
    incomplete_transaction = {
        "amt": 50.0,
    }

    response = client.post(
        "/predict",
        json=incomplete_transaction,
        headers=auth_headers(),
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Missing required fields"
    assert "fields" in data
    assert len(data["fields"]) > 0


def test_predict_invalid_data_type(client):
    """Prediction should reject invalid field types."""
    invalid_transaction = VALID_TRANSACTION.copy()
    invalid_transaction["amt"] = "fifty"

    response = client.post(
        "/predict",
        json=invalid_transaction,
        headers=auth_headers(),
    )

    assert response.status_code == 400

    data = response.get_json()

    assert data["error"] == "Invalid type for 'amt'"


def test_predict_invalid_datetime(client):
    """Prediction should reject an invalid datetime format."""
    invalid_transaction = VALID_TRANSACTION.copy()
    invalid_transaction["trans_date_trans_time"] = "2025-01-01 12:30"

    response = client.post(
        "/predict",
        json=invalid_transaction,
        headers=auth_headers(),
    )

    assert response.status_code == 400

    data = response.get_json()

    assert "Invalid trans_date_trans_time format" in data["error"]
