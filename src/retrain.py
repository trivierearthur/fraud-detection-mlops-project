from pathlib import Path
import subprocess
import sys

CURRENT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = CURRENT_DIR / "train_model.py"


def retrain_model() -> None:
    """Run the existing training pipeline to create a new model version."""

    print("Starting model retraining...")
    print("---------------------------")

    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Model retraining failed with exit code {result.returncode}."
        )

    print("\nModel retraining completed successfully.")


if __name__ == "__main__":
    retrain_model()
