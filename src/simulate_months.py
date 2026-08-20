from pathlib import Path

import numpy as np
import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

SOURCE_FILE = PROJECT_ROOT / "data" / "processed" / "fraud_data_cleaned.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "monthly"

RANDOM_STATE = 42
MONTHS = 12
ROWS_PER_MONTH = 1000


def load_source_data() -> pd.DataFrame:
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE_FILE}")

    return pd.read_csv(SOURCE_FILE)


def create_monthly_data(
    source: pd.DataFrame,
    month: int,
    rng: np.random.Generator,
) -> pd.DataFrame:

    # Sample transactions from the original dataset
    monthly = source.sample(
        n=ROWS_PER_MONTH,
        replace=True,
        random_state=RANDOM_STATE + month,
    ).copy()

    # ---------------------------------------------------------
    # Introduce simulated data drift
    # ---------------------------------------------------------

    if month == 4:
        # Higher transaction amounts
        monthly["amt"] = monthly["amt"] * 1.5

    elif month == 6:
        # More transactions during late-night hours
        if "trans_hour" in monthly.columns:
            monthly["trans_hour"] = rng.choice(
                [22, 23, 0, 1, 2, 3],
                size=len(monthly),
            )

    elif month == 8:
        # Change category distribution
        if "category" in monthly.columns:
            monthly["category"] = rng.choice(
                [
                    "shopping_net",
                    "grocery_pos",
                    "shopping_pos",
                    "gas_transport",
                ],
                size=len(monthly),
                p=[0.40, 0.30, 0.20, 0.10],
            )

    elif month == 10:
        # Stronger increase in transaction amounts
        monthly["amt"] = monthly["amt"] * 2.0

    elif month == 12:
        # Combined drift
        monthly["amt"] = monthly["amt"] * 1.8

        if "trans_hour" in monthly.columns:
            monthly["trans_hour"] = rng.choice(
                [21, 22, 23, 0, 1, 2, 3],
                size=len(monthly),
            )

    return monthly


def main() -> None:
    print("Creating monthly simulation data")
    print("================================")

    source = load_source_data()

    print(f"Source rows: {len(source):,}")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(RANDOM_STATE)

    for month in range(1, MONTHS + 1):
        monthly = create_monthly_data(
            source=source,
            month=month,
            rng=rng,
        )

        output_file = OUTPUT_DIR / f"month_{month:02d}.csv"

        monthly.to_csv(
            output_file,
            index=False,
        )

        print(f"Month {month:02d}: {len(monthly):,} rows → {output_file.name}")

    print()
    print("Monthly simulation completed.")


if __name__ == "__main__":
    main()
