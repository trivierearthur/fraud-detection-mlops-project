import os
from functools import wraps

from flask import jsonify, request

API_KEY = os.getenv("API_KEY")


def require_api_key(f):
    """Require a valid API key to access a protected endpoint."""

    @wraps(f)
    def decorated_function(*args, **kwargs):

        # Check whether the Authorization header exists
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            return jsonify({"error": "Authentication required"}), 401

        # Check the expected Bearer token format
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Invalid authentication format"}), 401

        # Extract the API key
        token = auth_header.split(" ", 1)[1]

        # Check that an API key has been configured
        if not API_KEY:
            return jsonify({"error": "API authentication is not configured"}), 500

        # Validate the API key
        if token != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

        return f(*args, **kwargs)

    return decorated_function
