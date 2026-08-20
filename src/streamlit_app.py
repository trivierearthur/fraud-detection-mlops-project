import os

import requests
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables
load_dotenv()

API_KEY = os.getenv("API_KEY")
API_URL = os.getenv("API_URL", "http://127.0.0.1:5000")

st.set_page_config(
    page_title="Fraud Detection",
    page_icon="🛡️",
    layout="centered",
)

st.title("🛡️ Fraud Detection")
st.caption("ML-powered transaction risk assessment")

st.divider()

st.subheader("Transaction details")

col1, col2 = st.columns(2)

with col1:
    amount = st.number_input(
        "Transaction amount",
        min_value=0.0,
        value=50.0,
        step=1.0,
    )

    category = st.text_input(
        "Category",
        value="grocery_pos",
    )

    merchant = st.text_input(
        "Merchant",
        value="Test Merchant",
    )

    city = st.text_input(
        "City",
        value="Test City",
    )

    state = st.text_input(
        "State",
        value="CA",
    )

    city_pop = st.number_input(
        "City population",
        min_value=0,
        value=100000,
        step=1000,
    )

    job = st.text_input(
        "Customer job",
        value="Analyst",
    )


with col2:
    transaction_datetime = st.text_input(
        "Transaction date/time",
        value="01-01-2025 12:30",
        help="Format: DD-MM-YYYY HH:MM",
    )

    dob = st.text_input(
        "Date of birth",
        value="01-01-1990",
        help="Format: DD-MM-YYYY",
    )

    latitude = st.number_input(
        "Customer latitude",
        value=40.0,
    )

    longitude = st.number_input(
        "Customer longitude",
        value=-75.0,
    )

    merchant_latitude = st.number_input(
        "Merchant latitude",
        value=40.1,
    )

    merchant_longitude = st.number_input(
        "Merchant longitude",
        value=-75.1,
    )

    transaction_number = st.text_input(
        "Transaction ID",
        value="DEMO123",
    )


st.divider()

predict_button = st.button(
    "🔍 Analyze Transaction",
    type="primary",
    use_container_width=True,
)


if predict_button:

    if not API_KEY:
        st.error("API key not configured.")
        st.stop()

    transaction = {
        "trans_date_trans_time": transaction_datetime,
        "merchant": merchant,
        "category": category,
        "amt": amount,
        "city": city,
        "state": state,
        "lat": latitude,
        "long": longitude,
        "city_pop": city_pop,
        "job": job,
        "dob": dob,
        "trans_num": transaction_number,
        "merch_lat": merchant_latitude,
        "merch_long": merchant_longitude,
    }

    headers = {
        "Authorization": f"Bearer {API_KEY}",
    }

    try:

        response = requests.post(
            f"{API_URL}/predict",
            json=transaction,
            headers=headers,
            timeout=10,
        )

        if response.status_code == 200:

            result = response.json()

            probability = result["fraud_probability"]
            prediction = result["fraud_prediction"]

            st.subheader("Prediction")

            metric_col1, metric_col2 = st.columns(2)

            with metric_col1:
                st.metric(
                    "Fraud probability",
                    f"{probability:.1%}",
                )

            with metric_col2:
                if prediction == 1:
                    st.metric(
                        "Risk assessment",
                        "FRAUD",
                    )
                else:
                    st.metric(
                        "Risk assessment",
                        "LEGITIMATE",
                    )

            if prediction == 1:
                st.error(
                    "⚠️ This transaction has been flagged as potentially fraudulent."
                )
            else:
                st.success("✅ This transaction has been classified as legitimate.")

        else:

            error_data = response.json()

            st.error(
                error_data.get(
                    "error",
                    f"API request failed with status {response.status_code}",
                )
            )

    except requests.exceptions.ConnectionError:

        st.error("Could not connect to the Flask API. " "Make sure api.py is running.")

    except requests.exceptions.Timeout:

        st.error("The API request timed out.")

    except Exception as error:

        st.error(f"Unexpected error: {error}")
