"""
Entry point for the OFFLINE / TRAINING pipeline (CI/CD side).

Flow:
    train_model.train() internally: data_loader -> preprocess -> model
    selection/training -> model versioning (fraud_model_vN.joblib).

For SERVING (api.py + auth.py + predict.py + monitoring.py),
use run_app.py instead — that's a long-running process, not a
one-shot pipeline step.

retrain.py reuses run_training_pipeline() when drift is detected
by monitoring.py, closing the CI/CD loop.
"""

# pyright: strict
import logging
from pathlib import Path

from src.train_model import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def run_training_pipeline() -> Path:
    """Runs the full training pipeline and returns the saved model path."""
    logger.info("=== Training pipeline started ===")

    model_path = train()

    logger.info("=== Training pipeline completed. Model saved at %s ===", model_path)
    return model_path


if __name__ == "__main__":
    run_training_pipeline()
