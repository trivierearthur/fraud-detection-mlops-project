from pathlib import Path
import sys

import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from src.monitoring import (
        load_reference_data,
        load_monthly_data,
        monitor_month,
    )
except ModuleNotFoundError:
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.append(str(PROJECT_ROOT))
    from src.monitoring import (
        load_reference_data,
        load_monthly_data,
        monitor_month,
    )

OUTPUT_FILE = PROJECT_ROOT / "data" / "monitoring_drift.png"


def main():
    reference = load_reference_data()

    months = []
    drift_counts = []

    for month in range(1, 13):
        current = load_monthly_data(month)

        result = monitor_month(
            reference=reference,
            current=current,
            month=month,
        )

        months.append(f"Month {month:02d}")
        drift_counts.append(len(result["drifted_features"]))

    plt.figure(figsize=(10, 5))

    plt.bar(months, drift_counts)

    plt.xlabel("Month")
    plt.ylabel("Number of drifted features")
    plt.title("Detected Data Drift Over 12 Simulated Months")

    plt.ylim(0, max(drift_counts) + 1)

    plt.xticks(rotation=45)
    plt.tight_layout()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_FILE, dpi=300)
    plt.show()

    print(f"\nGraph saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
