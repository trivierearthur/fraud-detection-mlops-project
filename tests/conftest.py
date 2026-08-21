"""Pytest configuration and fixtures for tests."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def ensure_api_key():
    """Ensure API_KEY is set from environment or use a fallback for testing."""
    if not os.getenv("API_KEY"):
        # Set a default test key if not provided
        os.environ["API_KEY"] = "test-api-key"
