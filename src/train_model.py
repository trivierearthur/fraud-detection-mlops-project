from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.data_loader import load_raw_data
from src.preprocess import (
    TransactionFeatureTransformer,
    engineer_transaction_features,
    prepare_training_dataframe,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    """Build column-wise preprocessing for numeric and categorical features."""

    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_pipeline(
    x: pd.DataFrame,
    estimator: Any,
) -> Pipeline:
    """Build complete feature-engineering, preprocessing and model pipeline."""

    engineered_x, _ = engineer_transaction_features(
        x,
        verbose=False,
    )

    preprocessor = build_preprocessor(engineered_x)

    return Pipeline(
        steps=[
            (
                "feature_engineering",
                TransactionFeatureTransformer(),
            ),
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "model",
                estimator,
            ),
        ]
    )


def get_models() -> dict[str, Any]:
    """Return candidate classification models."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "svc": CalibratedClassifierCV(
            estimator=SVC(
                probability=False,  # pyright: ignore[reportArgumentType] — sklearn stub types this as str, actual API takes bool
                class_weight="balanced",
                random_state=42,
            ),
            cv=3,
        ),
    }


def evaluate_model(
    pipeline: Pipeline,
    x: pd.DataFrame,
    y: pd.Series,
) -> dict[str, float]:
    """Evaluate a fitted pipeline."""

    probabilities = pipeline.predict_proba(x)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "roc_auc": roc_auc_score(y, probabilities),
        "pr_auc": average_precision_score(y, probabilities),
        "precision": precision_score(
            y,
            predictions,
            zero_division=0,  # pyright: ignore[reportArgumentType] — sklearn stub types this as str, actual API also takes 0/1
        ),
        "recall": recall_score(
            y,
            predictions,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
        "f1": f1_score(
            y,
            predictions,
            zero_division=0,  # pyright: ignore[reportArgumentType]
        ),
    }


def save_model(pipeline: Pipeline) -> Path:
    """
    Save a new model version.

    Versions are created sequentially:
    fraud_model_v1.joblib
    fraud_model_v2.joblib
    fraud_model_v3.joblib
    ...

    The active version is recorded in current_model.txt.
    """

    existing_versions = []

    for path in MODELS_DIR.glob("fraud_model_v*.joblib"):
        try:
            version = int(path.stem.split("_v")[-1])
            existing_versions.append(version)
        except ValueError:
            continue

    next_version = max(existing_versions) + 1 if existing_versions else 1

    model_filename = f"fraud_model_v{next_version}.joblib"

    model_path = MODELS_DIR / model_filename

    joblib.dump(
        pipeline,
        model_path,
    )

    current_model_file = MODELS_DIR / "current_model.txt"

    current_model_file.write_text(
        model_filename,
        encoding="utf-8",
    )

    print(f"\nModel saved successfully:\n  {model_path}\nCurrent model:\n  {model_filename}")

    return model_path


def report_top_feature_importance(
    best_result: dict[str, Any],
    top_n: int = 15,
) -> None:
    """Display top feature importance for Random Forest."""

    if best_result["model_name"] != "random_forest":
        return

    pipeline: Pipeline = best_result["pipeline"]
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return

    feature_names = preprocessor.get_feature_names_out()

    importances = model.feature_importances_

    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(
        by="importance",
        ascending=False,
    )

    print("\nTop feature importance")
    print("----------------------")

    print(feature_df.head(top_n).to_string(index=False))


def train() -> Path:
    """Train candidate models and save the best model."""

    print("=" * 50)
    print("Fraud Detection Model Training")
    print("=" * 50)

    # 1. Load raw data
    df = load_raw_data()

    print(f"\nRaw dataset shape: {df.shape}")

    # 2. Prepare training data
    x, y, summary = prepare_training_dataframe(df)

    print("\nTraining data summary:")

    for key, value in summary.items():
        print(f"  {key}: {value}")

    print(f"  fraud_count: {int(y.sum())}")

    print(f"  legitimate_count: {int((y == 0).sum())}")

    print(f"  fraud_rate: {y.mean():.2%}")

    # 3. Cross-validation
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    results = []

    print("\nTraining candidate models...")

    for model_name, estimator in get_models().items():
        print(f"\nModel: {model_name}")

        fold_scores = []

        for fold, (
            train_idx,
            validation_idx,
        ) in enumerate(
            cv.split(x, y),
            start=1,
        ):
            x_train = x.iloc[train_idx]
            x_validation = x.iloc[validation_idx]

            y_train = y.iloc[train_idx]
            y_validation = y.iloc[validation_idx]

            pipeline = build_pipeline(
                x_train,
                estimator,
            )

            pipeline.fit(
                x_train,
                y_train,
            )

            metrics = evaluate_model(
                pipeline,
                x_validation,
                y_validation,
            )

            fold_scores.append(metrics)

            print(
                f"  Fold {fold}: "
                f"ROC-AUC={metrics['roc_auc']:.4f}, "
                f"PR-AUC={metrics['pr_auc']:.4f}"
            )

        mean_metrics = {
            metric: float(np.mean([fold[metric] for fold in fold_scores]))
            for metric in fold_scores[0]
        }

        print(f"  Mean ROC-AUC: {mean_metrics['roc_auc']:.4f}")

        print(f"  Mean PR-AUC: {mean_metrics['pr_auc']:.4f}")

        results.append(
            {
                "model_name": model_name,
                **mean_metrics,
            }
        )

    # 4. Select best model using PR-AUC
    best_result = max(
        results,
        key=lambda result: result["pr_auc"],
    )

    best_model_name = best_result["model_name"]

    print("\nBest model:")
    print(f"  {best_model_name}")

    print(f"  PR-AUC: {best_result['pr_auc']:.4f}")

    print(f"  ROC-AUC: {best_result['roc_auc']:.4f}")

    # 5. Retrain selected model on all available data
    print("\nTraining final model on full dataset...")

    final_estimator = get_models()[best_model_name]

    final_pipeline = build_pipeline(
        x,
        final_estimator,
    )

    final_pipeline.fit(
        x,
        y,
    )

    # 6. Report feature importance
    report_top_feature_importance(
        {
            "model_name": best_model_name,
            "pipeline": final_pipeline,
        }
    )

    # 7. Save versioned model
    model_path = save_model(final_pipeline)

    print("\nTraining completed successfully.")

    return model_path


if __name__ == "__main__":
    train()
