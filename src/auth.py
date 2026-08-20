import os
from functools import wraps

from dotenv import load_dotenv
from flask import jsonify, request

load_dotenv()

API_KEY = os.getenv("API_KEY")


def require_api_key(function):
    @wraps(function)
    def decorated(*args, **kwargs):

        authorization = request.headers.get("Authorization")

        if not authorization:
            return jsonify({"error": "Authentication required"}), 401

        if not authorization.startswith("Bearer "):
            return jsonify({"error": "Invalid API key"}), 401

        token = authorization.split(" ", 1)[1]

        if token != API_KEY:
            return jsonify({"error": "Invalid API key"}), 401

        return function(*args, **kwargs)

    return decorated
