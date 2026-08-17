from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Reference data used to train the model
REFERENCE_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_data_cleaned.csv"

# Current/incoming data to monitor
CURRENT_FILE = (
    PROJECT_ROOT / "data" / "raw" / "synthetic_mixed_transactions_v3_nolabel.csv"
)

# Statistical significance threshold
SIGNIFICANCE_LEVEL = 0.05


# Features selected for monitoring
NUMERICAL_FEATURES = [
    "amt",
    "trans_hour",
    "trans_dayofweek",
    "customer_age",
    "merchant_distance_km",
    "city_pop",
]

CATEGORICAL_FEATURES = [
    "category",
    "state",
]


def load_data():
    """Load reference and current datasets."""

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_FILE}")

    if not CURRENT_FILE.exists():
        raise FileNotFoundError(f"Current data not found: {CURRENT_FILE}")

    reference = pd.read_csv(REFERENCE_FILE)
    current = pd.read_csv(CURRENT_FILE)

    return reference, current


def check_numerical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """Check numerical features using the Kolmogorov-Smirnov test."""

    results = []

    for feature in NUMERICAL_FEATURES:

        if feature not in reference.columns:
            print(f"Warning: {feature} is missing from reference data. " "Skipping.")
            continue

        if feature not in current.columns:
            print(f"Warning: {feature} is missing from current data. " "Skipping.")
            continue

        reference_values = pd.to_numeric(
            reference[feature],
            errors="coerce",
        ).dropna()

        current_values = pd.to_numeric(
            current[feature],
            errors="coerce",
        ).dropna()

        if len(reference_values) == 0 or len(current_values) == 0:
            print(f"Warning: {feature} contains no usable values. " "Skipping.")
            continue

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        results.append(
            {
                "feature": feature,
                "test": "KS",
                "statistic": statistic,
                "p_value": p_value,
                "drift": p_value < SIGNIFICANCE_LEVEL,
            }
        )

    return results


def check_categorical_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
):
    """Check categorical features using a chi-square test."""

    results = []

    for feature in CATEGORICAL_FEATURES:

        if feature not in reference.columns:
            print(f"Warning: {feature} is missing from reference data. " "Skipping.")
            continue

        if feature not in current.columns:
            print(f"Warning: {feature} is missing from current data. " "Skipping.")
            continue

        reference_counts = reference[feature].value_counts()
        current_counts = current[feature].value_counts()

        categories = reference_counts.index.union(current_counts.index)

        reference_distribution = reference_counts.reindex(
            categories,
            fill_value=0,
        )

        current_distribution = current_counts.reindex(
            categories,
            fill_value=0,
        )

        contingency_table = pd.DataFrame(
            [
                reference_distribution.values,
                current_distribution.values,
            ]
        )

        chi2, p_value, _, _ = chi2_contingency(contingency_table)

        results.append(
            {
                "feature": feature,
                "test": "Chi-square",
                "statistic": chi2,
                "p_value": p_value,
                "drift": p_value < SIGNIFICANCE_LEVEL,
            }
        )

    return results


def print_results(results):
    """Print drift results in a readable format."""

    print("\nDrift results")
    print("-------------")

    for result in results:

        status = "DRIFT" if result["drift"] else "OK"

        print(
            f"{result['feature']:<25}"
            f"{result['test']:<12}"
            f"p={result['p_value']:.4f}  "
            f"{status}"
        )


def main():

    print("Data Drift Monitoring")
    print("=====================")

    reference, current = load_data()

    print(f"Reference rows: {len(reference):,}")
    print(f"Current rows:   {len(current):,}")

    results = []

    numerical_results = check_numerical_drift(
        reference,
        current,
    )

    categorical_results = check_categorical_drift(
        reference,
        current,
    )

    results.extend(numerical_results)
    results.extend(categorical_results)

    print_results(results)

    drift_detected = any(result["drift"] for result in results)

    print("\nOverall result")
    print("--------------")

    if drift_detected:
        print("DRIFT DETECTED")
        print("Model retraining should be considered.")
    else:
        print("NO SIGNIFICANT DRIFT DETECTED")
        print("No retraining required.")


if __name__ == "__main__":
    main()
