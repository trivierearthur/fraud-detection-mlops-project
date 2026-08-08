from pathlib import Path
import sys

import pandas as pd

# Determine the project root so imports work when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load the local raw dataset from the project data folder.
from src.data_loader import load_raw_data

# Optional plotting support for EDA visuals.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def basic_eda(df: pd.DataFrame) -> None:
    # Main entry point for the exploratory analysis.
    print("=" * 80)
    print("FRAUD DATASET EXPLORATORY ANALYSIS")
    print("=" * 80)

    # Basic dataset overview.
    print("\n1) Dataset shape")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    # Show the type of each column to understand data quality and encoding.
    print("\n2) Columns and dtypes")
    print(df.dtypes.to_string())

    # Check whether there are missing values and quantify them.
    print("\n3) Missing values")
    missing = df.isna().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_summary = pd.DataFrame(
        {"missing_count": missing, "missing_pct": missing_pct}
    ).sort_values(["missing_count", "missing_pct"], ascending=False)
    print(missing_summary[missing_summary["missing_count"] > 0].to_string())
    if (missing_summary["missing_count"] == 0).all():
        print("No missing values found.")

    # Identify duplicate rows because they can distort analysis and training.
    print("\n4) Duplicate rows")
    duplicate_count = int(df.duplicated().sum())
    print(f"Duplicate rows: {duplicate_count}")

    # Summarize the numeric columns with descriptive statistics.
    print("\n5) Numeric summary")
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        print(df[numeric_cols].describe().T.to_string())
    else:
        print("No numeric columns found.")

    # Inspect the categorical/object columns and their most common values.
    print("\n6) Categorical / object summary")
    categorical_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
    if categorical_cols:
        for col in categorical_cols:
            print(f"\n{col}:")
            print(df[col].value_counts(dropna=False).head(10).to_string())
    else:
        print("No categorical/object columns found.")

    # Show a few example rows to confirm the data looks as expected.
    print("\n7) Sample rows")
    print(df.head().to_string())

    # Explore time-based fraud behavior.
    print("\n8) Temporal analysis")
    temporal_summary(df)

    # Examine relationships between numeric variables.
    print("\n9) Correlation analysis")
    correlation_summary(df)

    # Highlight class imbalance, which is critical for model selection and evaluation.
    print("\n10) Target distribution")
    target_summary = target_distribution_summary(df)
    print(target_summary.to_string())

    # Generate plots that highlight the most meaningful patterns.
    print("\n11) Visual summaries")
    create_visuals(df)


def temporal_summary(df: pd.DataFrame) -> None:
    # Convert the timestamp column and extract useful time-based features.
    df = df.copy()
    time_col = "trans_date_trans_time"
    if time_col not in df.columns:
        print(f"Column {time_col} not found; skipping temporal analysis.")
        return

    try:
        df[time_col] = pd.to_datetime(
            df[time_col], format="%d-%m-%Y %H:%M", errors="coerce"
        )
    except Exception:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Extract hourly, daily, weekday, monthly, and weekend features.
    df["hour"] = df[time_col].dt.hour
    df["day"] = df[time_col].dt.day
    df["weekday"] = df[time_col].dt.day_name()
    df["month"] = df[time_col].dt.month_name()
    df["is_weekend"] = df[time_col].dt.dayofweek >= 5

    # Clean the target column so it can be used consistently in calculations.
    fraud_col = "is_fraud"
    if fraud_col in df.columns:
        df["is_fraud_clean"] = (
            df[fraud_col].astype(str).str.extract(r"([01])", expand=False).astype(float)
        )
    else:
        df["is_fraud_clean"] = pd.NA

    print("\nTemporal fraud analysis")
    # Show the average fraud rate by hour, weekday, and month.
    print("Fraud by hour")
    print(
        df.groupby("hour")["is_fraud_clean"]
        .mean()
        .sort_values(ascending=False)
        .round(4)
        .to_string()
    )

    print("\nFraud by weekday")
    print(
        df.groupby("weekday")["is_fraud_clean"]
        .mean()
        .reindex(
            [
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            ]
        )
        .round(4)
        .to_string()
    )

    print("\nFraud by month")
    print(df.groupby("month")["is_fraud_clean"].mean().round(4).to_string())

    print("\nTransaction volume over time (daily)")
    daily_volume = df.groupby(df[time_col].dt.date).size()
    print(daily_volume.head(10).to_string())


def correlation_summary(df: pd.DataFrame) -> None:
    # Focus only on numeric columns for the correlation matrix.
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        print("Not enough numeric columns for correlation analysis.")
        return

    corr = numeric_df.corr(numeric_only=True)
    print("\nCorrelation matrix")
    print(corr.round(3).to_string())


def target_distribution_summary(df: pd.DataFrame) -> pd.DataFrame:
    # Show the class balance for the target label to inform model choice.
    target_counts = df["is_fraud"].value_counts()
    target_pct = df["is_fraud"].value_counts(normalize=True) * 100

    target_summary = pd.DataFrame(
        {"count": target_counts, "percentage": target_pct.round(2)}
    )
    return target_summary


def create_visuals(df: pd.DataFrame) -> None:
    # Create a few charts that make the key patterns easier to interpret.
    if plt is None:
        print("matplotlib is not available; skipping plots.")
        return

    df = df.copy()
    time_col = "trans_date_trans_time"
    if time_col in df.columns:
        try:
            df[time_col] = pd.to_datetime(
                df[time_col], format="%d-%m-%Y %H:%M", errors="coerce"
            )
        except Exception:
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df["hour"] = df[time_col].dt.hour

    df["is_fraud_clean"] = (
        df["is_fraud"].astype(str).str.extract(r"([01])", expand=False).astype(float)
    )

    # Save plots into a dedicated folder for easy inspection.
    plots_dir = PROJECT_ROOT / "notebooks" / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot fraud rate by transaction category.
    category_rate = (
        df.groupby("category")["is_fraud_clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    category_rate.plot(kind="bar", ax=ax, color="tomato")
    ax.set_title("Top 10 categories by fraud rate")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("Category")
    fig.tight_layout()
    fig.savefig(plots_dir / "fraud_rate_by_category.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot fraud rate by state.
    state_rate = (
        df.groupby("state")["is_fraud_clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    state_rate.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 10 states by fraud rate")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("State")
    fig.tight_layout()
    fig.savefig(plots_dir / "fraud_rate_by_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the class distribution to emphasize imbalance.
    target_counts = df["is_fraud"].astype(str).value_counts()
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Target class distribution")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Compare the amount distribution for legitimate vs fraudulent transactions.
    legit_amounts = df.loc[df["is_fraud_clean"] == 0, "amt"].dropna()
    fraud_amounts = df.loc[df["is_fraud_clean"] == 1, "amt"].dropna()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(legit_amounts, bins=40, alpha=0.6, label="Legitimate", color="skyblue")
    ax.hist(fraud_amounts, bins=40, alpha=0.7, label="Fraud", color="tomato")
    ax.set_title("Transaction amount distribution by label")
    ax.set_xlabel("Amount")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        plots_dir / "transaction_amount_distribution.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Visualize fraud rate by hour of day.
    if "hour" in df.columns:
        hourly_rate = df.groupby("hour")["is_fraud_clean"].mean()
        fig, ax = plt.subplots(figsize=(10, 5))
        hourly_rate.plot(kind="line", marker="o", ax=ax, color="purple")
        ax.set_title("Fraud rate by hour of day")
        ax.set_ylabel("Fraud rate")
        ax.set_xlabel("Hour")
        fig.tight_layout()
        fig.savefig(plots_dir / "fraud_rate_by_hour.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    # Plot transaction volume over time to identify spikes or seasonal behavior.
    if "trans_date_trans_time" in df.columns:
        daily_volume = df.groupby(df[time_col].dt.date).size()
        fig, ax = plt.subplots(figsize=(10, 5))
        daily_volume.plot(kind="line", ax=ax, color="green")
        ax.set_title("Transaction volume over time")
        ax.set_ylabel("Transactions")
        ax.set_xlabel("Date")
        fig.tight_layout()
        fig.savefig(
            plots_dir / "transaction_volume_over_time.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    print(f"Saved plots to {plots_dir}")


if __name__ == "__main__":
    # Load the data and run the full exploratory analysis pipeline.
    df = load_raw_data()
    basic_eda(df)
