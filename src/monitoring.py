from pathlib import Path

import pandas as pd
from scipy.stats import chi2_contingency, ks_2samp

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

# Reference dataset used to train the model
REFERENCE_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_data_cleaned.csv"

# Directory containing simulated monthly data
MONTHLY_DIR = PROJECT_ROOT / "data" / "monthly"

SIGNIFICANCE_LEVEL = 0.05


# Numerical features monitored for drift
NUMERICAL_FEATURES = [
    "amt",
    "trans_hour",
    "trans_dayofweek",
    "customer_age",
    "merchant_distance_km",
    "city_pop",
]


# Categorical features monitored for drift
CATEGORICAL_FEATURES = [
    "category",
    "state",
]


def load_reference_data():
    """Load the reference dataset."""

    if not REFERENCE_FILE.exists():
        raise FileNotFoundError(f"Reference data not found: {REFERENCE_FILE}")

    return pd.read_csv(REFERENCE_FILE)


def load_monthly_data(month: int):
    """Load one simulated month's data."""

    month_file = MONTHLY_DIR / f"month_{month:02d}.csv"

    if not month_file.exists():
        raise FileNotFoundError(f"Monthly data not found: {month_file}")

    return pd.read_csv(month_file)


def check_numerical_drift(reference, current):
    """Use the Kolmogorov-Smirnov test for numerical features."""

    results = []

    for feature in NUMERICAL_FEATURES:

        if feature not in reference.columns:
            continue

        if feature not in current.columns:
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


def check_categorical_drift(reference, current):
    """Use a chi-square test for categorical features."""

    results = []

    for feature in CATEGORICAL_FEATURES:

        if feature not in reference.columns:
            continue

        if feature not in current.columns:
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


def monitor_month(reference, current, month):
    """Run all drift tests for one month."""

    numerical_results = check_numerical_drift(
        reference,
        current,
    )

    categorical_results = check_categorical_drift(
        reference,
        current,
    )

    results = numerical_results + categorical_results

    drifted_features = [result["feature"] for result in results if result["drift"]]

    drift_detected = len(drifted_features) > 0

    print(f"\nMonth {month:02d}")
    print("-------")

    for result in results:

        status = "DRIFT" if result["drift"] else "OK"

        print(
            f"{result['feature']:<25}"
            f"{result['test']:<12}"
            f"p={result['p_value']:.4f}  "
            f"{status}"
        )

    if drift_detected:
        print(f"RESULT: DRIFT DETECTED " f"({len(drifted_features)} feature(s))")

        print("Action: model retraining should be triggered.")

    else:
        print("RESULT: NO SIGNIFICANT DRIFT")

        print("Action: keep the current model.")

    return {
        "month": month,
        "drift_detected": drift_detected,
        "drifted_features": drifted_features,
    }


def main():

    print("Data Drift Monitoring")
    print("=====================")

    reference = load_reference_data()

    print(f"Reference dataset: " f"{len(reference):,} rows")

    monthly_results = []

    for month in range(1, 13):

        current = load_monthly_data(month)

        result = monitor_month(
            reference=reference,
            current=current,
            month=month,
        )

        monthly_results.append(result)

    print("\n")
    print("Annual Drift Summary")
    print("====================")

    for result in monthly_results:

        if result["drift_detected"]:

            features = ", ".join(result["drifted_features"])

            print(f"Month {result['month']:02d}: " f"DRIFT → {features}")

        else:

            print(f"Month {result['month']:02d}: " f"NO DRIFT")

    drift_months = [
        result["month"] for result in monthly_results if result["drift_detected"]
    ]

    print()
    print(f"Months with detected drift: " f"{len(drift_months)}/12")

    if drift_months:

        print(
            "Retraining would be triggered for: "
            + ", ".join(f"month {month:02d}" for month in drift_months)
        )

    else:

        print("No retraining would be triggered.")


if __name__ == "__main__":
    main()
