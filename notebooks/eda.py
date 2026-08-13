from pathlib import Path
import sys

import pandas as pd

# Determine the project root so imports work when this script is run directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Load the local raw dataset from the project data folder.
try:
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    from fraud_detection_mlops_project.src.data_loader import load_raw_data

# Optional plotting support for EDA visuals.
try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def clean_is_fraud(series: pd.Series) -> pd.Series:
    """Normalize the target column to a numeric 0/1 series."""
    return pd.to_numeric(series, errors="coerce").fillna(0).clip(0, 1)


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

    # Analyze age distribution for fraud cases.
    print("\n10) Fraud analysis by customer age")
    age_fraud_summary(df)

    # Analyze city-level fraud concentration and risk.
    print("\n11) Fraud analysis by city")
    city_fraud_summary(df)

    # Generate plots that highlight the most meaningful patterns.
    print("\n12) Visual summaries")
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
    # Show the total number of fraud cases by hour, weekday, and month.
    print("Fraud by hour")
    print(
        df.groupby("hour")["is_fraud_clean"]
        .sum()
        .sort_values(ascending=False)
        .to_string()
    )

    print("\nFraud by weekday")
    print(
        df.groupby("weekday")["is_fraud_clean"]
        .sum()
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
        .to_string()
    )

    print("\nFraud by month")
    print(df.groupby("month")["is_fraud_clean"].sum().to_string())

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


def age_fraud_summary(df: pd.DataFrame) -> None:
    df = df.copy()
    if "dob" not in df.columns or "trans_date_trans_time" not in df.columns:
        print("Missing dob or trans_date_trans_time; skipping age analysis.")
        return

    df["is_fraud_clean"] = clean_is_fraud(df["is_fraud"])
    df["dob"] = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")
    df["trans_date_trans_time"] = pd.to_datetime(
        df["trans_date_trans_time"], dayfirst=True, errors="coerce"
    )
    df["customer_age"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["customer_age"] = df["customer_age"].where(df["customer_age"].between(0, 110))

    fraud_age = df.loc[df["is_fraud_clean"] == 1, "customer_age"].dropna()
    nonfraud_age = df.loc[df["is_fraud_clean"] == 0, "customer_age"].dropna()

    if fraud_age.empty:
        print("No valid fraud age records found.")
        return

    print("Fraud customer age summary")
    print(f"Fraud records with valid age: {len(fraud_age)}")
    print(f"Mean age (fraud): {fraud_age.mean():.2f}")
    print(f"Median age (fraud): {fraud_age.median():.2f}")
    print(f"Mean age (non-fraud): {nonfraud_age.mean():.2f}")
    print(f"Median age (non-fraud): {nonfraud_age.median():.2f}")

    bins = [0, 25, 35, 45, 55, 65, 110]
    labels = ["0-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    df["age_band"] = pd.cut(df["customer_age"], bins=bins, labels=labels, right=False)

    age_band_counts = (
        df.loc[df["is_fraud_clean"] == 1, "age_band"]
        .value_counts(dropna=False)
        .reindex(labels)
        .fillna(0)
    )
    print("\nFraud counts by age band")
    print(age_band_counts.astype(int).to_string())


def city_fraud_summary(df: pd.DataFrame) -> None:
    df = df.copy()
    if "city" not in df.columns:
        print("Column city not found; skipping city analysis.")
        return

    df["is_fraud_clean"] = clean_is_fraud(df["is_fraud"])
    city_summary = (
        df.groupby("city", dropna=False)
        .agg(
            total_transactions=("is_fraud_clean", "size"),
            fraud_count=("is_fraud_clean", "sum"),
        )
        .reset_index()
    )
    city_summary["fraud_rate_pct"] = (
        city_summary["fraud_count"] / city_summary["total_transactions"] * 100
    )

    top_fraud_count = city_summary.sort_values("fraud_count", ascending=False).head(10)
    print("Top 10 cities by fraud count")
    print(top_fraud_count.to_string(index=False))

    stable_cities = city_summary[city_summary["total_transactions"] >= 20]
    top_fraud_rate = stable_cities.sort_values("fraud_rate_pct", ascending=False).head(
        10
    )
    print("\nTop 10 cities by fraud rate (min 20 transactions)")
    print(top_fraud_rate.to_string(index=False))


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

    df["is_fraud_clean"] = clean_is_fraud(df["is_fraud"])

    # Save plots into a dedicated folder for easy inspection.
    plots_dir = PROJECT_ROOT / "notebooks" / "eda_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot fraud rate by transaction category and annotate the total fraud cases.
    category_rate = (
        df.groupby("category")["is_fraud_clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    category_total = (
        df.groupby("category")["is_fraud_clean"].sum().reindex(category_rate.index)
    )
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = category_rate.plot(kind="bar", ax=ax, color="tomato")
    ax.set_title("Top 10 categories by fraud rate (with total fraud cases)")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("Category")
    for bar, total in zip(bars.patches, category_total.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(total)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "fraud_rate_by_category.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot fraud rate by state and annotate the total fraud cases.
    state_rate = (
        df.groupby("state")["is_fraud_clean"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    state_total = df.groupby("state")["is_fraud_clean"].sum().reindex(state_rate.index)
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = state_rate.plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title("Top 10 states by fraud rate (with total fraud cases)")
    ax.set_ylabel("Fraud rate")
    ax.set_xlabel("State")
    for bar, total in zip(bars.patches, state_total.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(total)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )
    fig.tight_layout()
    fig.savefig(plots_dir / "fraud_rate_by_state.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Plot the class distribution to emphasize imbalance.
    target_counts = df["is_fraud_clean"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    target_counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Fraud vs non-fraud transaction counts")
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["Non-fraud (0)", "Fraud (1)"], rotation=0)
    fig.tight_layout()
    fig.savefig(
        plots_dir / "fraud_transaction_counts.png", dpi=300, bbox_inches="tight"
    )
    plt.close(fig)

    # Plot customer age distribution for fraud vs non-fraud.
    if "dob" in df.columns and "trans_date_trans_time" in df.columns:
        dob = pd.to_datetime(df["dob"], dayfirst=True, errors="coerce")
        trans_time = pd.to_datetime(
            df["trans_date_trans_time"], dayfirst=True, errors="coerce"
        )
        df["customer_age"] = (trans_time - dob).dt.days / 365.25
        df["customer_age"] = df["customer_age"].where(
            df["customer_age"].between(0, 110)
        )

        fraud_age = df.loc[df["is_fraud_clean"] == 1, "customer_age"].dropna()
        nonfraud_age = df.loc[df["is_fraud_clean"] == 0, "customer_age"].dropna()

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(nonfraud_age, bins=30, alpha=0.6, label="Non-fraud", color="skyblue")
        ax.hist(fraud_age, bins=30, alpha=0.7, label="Fraud", color="tomato")
        ax.set_title("Customer age distribution: fraud vs non-fraud")
        ax.set_xlabel("Customer age")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            plots_dir / "fraud_age_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)

    # Plot top cities by fraud count.
    if "city" in df.columns:
        city_fraud = (
            df.groupby("city")["is_fraud_clean"]
            .sum()
            .sort_values(ascending=False)
            .head(35)
        )
        city_fraud = city_fraud.sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(12, 11))
        city_fraud.plot(kind="barh", ax=ax, color="indianred")
        ax.set_title("Top 35 cities by number of fraud transactions")
        ax.set_xlabel("Fraud transaction count")
        ax.set_ylabel("City")
        fig.tight_layout()
        fig.savefig(plots_dir / "fraud_by_city_count.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Plot top cities by fraud rate with a minimum transaction threshold
        # to avoid noisy rates from very small sample sizes.
        city_summary = (
            df.groupby("city", dropna=False)["is_fraud_clean"]
            .agg(total_transactions="size", fraud_count="sum")
            .reset_index()
        )
        city_summary["nonfraud_count"] = (
            city_summary["total_transactions"] - city_summary["fraud_count"]
        )
        city_summary = city_summary[city_summary["total_transactions"] >= 20]
        city_summary["fraud_rate_pct"] = (
            city_summary["fraud_count"] / city_summary["total_transactions"] * 100
        )
        top_city_rate = city_summary.sort_values(
            "fraud_rate_pct", ascending=False
        ).head(35)
        top_city_rate = top_city_rate.sort_values("fraud_rate_pct", ascending=True)

        fig, ax = plt.subplots(figsize=(13, 12))
        top_city_rate.set_index("city")["fraud_rate_pct"].plot(
            kind="barh", ax=ax, color="darkorange"
        )
        ax.set_title("Top 35 cities by fraud rate (min 20 transactions)")
        ax.set_xlabel("Fraud rate (%)")
        ax.set_ylabel("City")

        # Add per-city volume context at the end of each bar.
        rate_max = (
            float(top_city_rate["fraud_rate_pct"].max())
            if not top_city_rate.empty
            else 0
        )
        ax.set_xlim(0, rate_max * 1.35 if rate_max > 0 else 1)
        for bar, (_, row) in zip(ax.patches, top_city_rate.iterrows()):
            label = (
                f"T:{int(row['total_transactions'])} "
                f"F:{int(row['fraud_count'])} "
                f"NF:{int(row['nonfraud_count'])}"
            )
            ax.text(
                bar.get_width() + rate_max * 0.01,
                bar.get_y() + bar.get_height() / 2,
                label,
                ha="left",
                va="center",
                fontsize=7,
            )

        fig.tight_layout()
        fig.savefig(plots_dir / "fraud_by_city_rate.png", dpi=300, bbox_inches="tight")
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

    # Visualize fraud rate by hour of day and annotate the total fraud cases.
    if "hour" in df.columns:
        hourly_rate = df.groupby("hour")["is_fraud_clean"].mean()
        hourly_counts = df.groupby("hour")["is_fraud_clean"].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        hourly_rate.plot(kind="line", marker="o", ax=ax, color="purple")
        ax.set_title("Fraud rate by hour of day (with total fraud cases)")
        ax.set_ylabel("Fraud rate")
        ax.set_xlabel("Hour")
        for x, y in zip(hourly_rate.index, hourly_rate.values):
            ax.text(
                x,
                y,
                f"{int(hourly_counts.loc[x])}",
                ha="left",
                va="bottom",
                fontsize=8,
                color="black",
            )
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
