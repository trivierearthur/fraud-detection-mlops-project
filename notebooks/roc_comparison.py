"""
Generate ROC and Precision-Recall curve comparison plots for the candidate
models used in src/train_model.py (logistic regression, random forest, SVC).
PR-AUC is included because it is the metric used to select the best model.

Run with: python notebooks/roc_comparison.py
Output: training_output/roc_comparison.png, training_output/pr_comparison.png
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_loader import load_raw_data
from src.preprocess import prepare_training_dataframe
from src.train_model import build_pipeline, get_models

OUTPUT_DIR = PROJECT_ROOT / "training_output"
ROC_OUTPUT_FILE = OUTPUT_DIR / "roc_comparison.png"
PR_OUTPUT_FILE = OUTPUT_DIR / "pr_comparison.png"


def main() -> None:
    df = load_raw_data()
    x, y, _ = prepare_training_dataframe(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, stratify=y, random_state=42
    )

    roc_fig, roc_ax = plt.subplots(figsize=(7, 6))
    pr_fig, pr_ax = plt.subplots(figsize=(7, 6))

    for model_name, estimator in get_models().items():
        pipeline = build_pipeline(x_train, estimator)
        pipeline.fit(x_train, y_train)

        probabilities = pipeline.predict_proba(x_test)[:, 1]

        RocCurveDisplay.from_predictions(
            y_test,
            probabilities,
            name=model_name,
            ax=roc_ax,
        )

        PrecisionRecallDisplay.from_predictions(
            y_test,
            probabilities,
            name=model_name,
            ax=pr_ax,
        )

    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    roc_ax.set_title("ROC Curve Comparison Across Candidate Models")
    roc_ax.legend(loc="lower right")

    fraud_rate = y_test.mean()
    pr_ax.axhline(fraud_rate, linestyle="--", color="gray", label="Chance")
    pr_ax.set_title("Precision-Recall Curve Comparison Across Candidate Models")
    pr_ax.legend(loc="upper right")
    # A curve staying near precision=1.0 until a late drop indicates strong performance;
    # this "flat-then-drop" shape is expected for imbalanced fraud data, not an artifact.
    pr_fig.text(
        0.5, -0.02,
        "PR-AUC (average precision) is used for model selection because ROC-AUC "
        "can look inflated on imbalanced fraud data.",
        ha="center", fontsize=8, color="dimgray",
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    roc_fig.savefig(ROC_OUTPUT_FILE, dpi=150, bbox_inches="tight")
    pr_fig.savefig(PR_OUTPUT_FILE, dpi=150, bbox_inches="tight")
    print(f"Saved ROC comparison plot to {ROC_OUTPUT_FILE}")
    print(f"Saved PR comparison plot to {PR_OUTPUT_FILE}")


if __name__ == "__main__":
    main()

