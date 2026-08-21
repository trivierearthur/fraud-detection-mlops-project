"""
Generate a ROC curve comparison plot for the candidate models used in
src/train_model.py (logistic regression, random forest, SVC).

Run with: python notebooks/roc_comparison.py
Output: training_output/roc_comparison.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.preprocess import prepare_training_dataframe
from src.train_model import build_pipeline, get_models

OUTPUT_DIR = PROJECT_ROOT / "training_output"
OUTPUT_FILE = OUTPUT_DIR / "roc_comparison.png"


def main() -> None:
    df = load_raw_data()
    x, y, _ = prepare_training_dataframe(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, estimator in get_models().items():
        pipeline = build_pipeline(x_train, estimator)
        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(x_test)[:, 1]

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=model_name,
            ax=ax,
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_title("ROC Curve Comparison Across Candidate Models")
    ax.legend(loc="lower right")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved ROC comparison plot to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
