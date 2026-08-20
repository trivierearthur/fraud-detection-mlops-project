from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from src.preprocess import (
        TransactionFeatureTransformer,
        engineer_transaction_features,
        prepare_training_dataframe,
    )
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    try:
        from fraud_detection_mlops_project.src.preprocess import (
            TransactionFeatureTransformer,
            engineer_transaction_features,
            prepare_training_dataframe,
        )
        from fraud_detection_mlops_project.src.data_loader import load_raw_data
    except ModuleNotFoundError:
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from preprocess import (
            TransactionFeatureTransformer,
            engineer_transaction_features,
            prepare_training_dataframe,
        )
        from data_loader import load_raw_data


MODELS_DIR = PROJECT_ROOT / "models"
MODEL_FILE = MODELS_DIR / "fraud_model.joblib"


def build_preprocessor(
    x: pd.DataFrame,
    scale_numeric: bool = False,
) -> ColumnTransformer:
    """Build column-wise preprocessing for numeric and categorical features."""
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_transformer = Pipeline(steps=numeric_steps)
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=20,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_model_pipeline(
    x: pd.DataFrame,
    estimator: Any,
    scale_numeric: bool = False,
) -> Pipeline:
    """Build end-to-end pipeline from raw transactions to predictions."""
    # Fit the column preprocessor against engineered columns so inference
    # and training share the exact same schema.
    engineered_x, _ = engineer_transaction_features(x, verbose=False)
    preprocessor = build_preprocessor(
        engineered_x,
        scale_numeric=scale_numeric,
    )
    return Pipeline(
        steps=[
            ("feature_engineering", TransactionFeatureTransformer()),
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def get_model_candidates() -> dict[str, tuple[Any, bool]]:
    """Return candidate estimators and whether numeric scaling is required."""
    return {
        "logistic_regression": (
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=42,
            ),
            True,
        ),
        "random_forest": (
            RandomForestClassifier(
                n_estimators=300,
                random_state=42,
                n_jobs=-1,
                class_weight="balanced_subsample",
            ),
            False,
        ),
        "svm": (
            CalibratedClassifierCV(
                estimator=SVC(
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=42,
                ),
                cv=3,
                ensemble=False,
            ),
            True,
        ),
    }


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, Any]:
    """Train one candidate, print holdout metrics, and return summary stats."""
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_score)
    pr_auc = average_precision_score(y_test, y_score)

    print(f"\n{name}")
    print("-" * len(name))
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC: {pr_auc:.4f}")

    return {
        "model_name": name,
        "pipeline": pipeline,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
    }


def save_model(pipeline: Pipeline) -> None:
    """Persist the fitted pipeline with a simple model version."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Find existing model versions
    existing_models = list(MODELS_DIR.glob("fraud_model_v*.joblib"))

    if existing_models:
        versions = []
        for model_path in existing_models:
            try:
                version = int(model_path.stem.replace("fraud_model_v", ""))
                versions.append(version)
            except ValueError:
                continue

        next_version = max(versions, default=0) + 1
    else:
        next_version = 1

    model_file = MODELS_DIR / f"fraud_model_v{next_version}.joblib"

    joblib.dump(pipeline, model_file)

    # Keep track of the currently active model
    current_model_file = MODELS_DIR / "current_model.txt"
    current_model_file.write_text(
        model_file.name,
        encoding="utf-8",
    )

    print(f"Saved model version: v{next_version}")
    print(f"Model path: {model_file}")
    print(f"Current model: {current_model_file}")


def check_for_leakage(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> None:
    """Run lightweight checks for obvious leakage and split contamination."""
    engineered_train, _ = engineer_transaction_features(x_train, verbose=False)
    suspicious_terms = ("fraud", "target", "label")
    suspicious_columns = [
        column
        for column in engineered_train.columns
        if any(term in column.lower() for term in suspicious_terms)
    ]

    print("\nLeakage checks")
    print("--------------")
    if suspicious_columns:
        print("Potentially suspicious feature names detected:")
        for column in suspicious_columns:
            print(f"  - {column}")
    else:
        print("No target-like feature names detected in the model inputs.")

    train_hashes = set(pd.util.hash_pandas_object(x_train, index=False).tolist())
    test_hashes = set(pd.util.hash_pandas_object(x_test, index=False).tolist())
    overlapping_rows = len(train_hashes & test_hashes)

    print(f"Exact feature-row overlap between train/test: {overlapping_rows}")
    if overlapping_rows > 0:
        print("Warning: duplicate feature rows exist across the split.")
    else:
        print("No exact feature-row overlap detected across the split.")


def validate_selected_model(
    best_result: dict[str, Any],
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Validate selected model with CV and a shuffled-target sanity test."""
    pipeline = best_result["pipeline"]
    baseline_pr_auc = y_test.mean()

    print("\nValidation on selected model")
    print("----------------------------")
    print(f"Selected model: {best_result['model_name']}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_validate(
        clone(pipeline),
        x_train,
        y_train,
        cv=cv,
        scoring={
            "pr_auc": "average_precision",
            "roc_auc": "roc_auc",
        },
        n_jobs=-1,
    )

    print("5-fold cross-validation on training split:")
    print(
        "  PR-AUC : "
        f"{cv_scores['test_pr_auc'].mean():.4f} "
        f"+/- {cv_scores['test_pr_auc'].std():.4f}"
    )
    print(
        "  ROC-AUC: "
        f"{cv_scores['test_roc_auc'].mean():.4f} "
        f"+/- {cv_scores['test_roc_auc'].std():.4f}"
    )

    shuffled_pipeline = clone(pipeline)
    rng = np.random.default_rng(42)
    shuffled_y_train = rng.permutation(y_train.to_numpy())
    shuffled_pipeline.fit(x_train, shuffled_y_train)
    shuffled_y_score = shuffled_pipeline.predict_proba(x_test)[:, 1]
    shuffled_pr_auc = average_precision_score(y_test, shuffled_y_score)
    shuffled_roc_auc = roc_auc_score(y_test, shuffled_y_score)
    pr_auc_delta = shuffled_pr_auc - baseline_pr_auc

    print("Shuffled-target sanity check:")
    print(f"  Baseline PR-AUC (fraud prevalence): {baseline_pr_auc:.4f}")
    print(f"  Shuffled PR-AUC                  : {shuffled_pr_auc:.4f}")
    print(f"  Shuffled ROC-AUC                 : {shuffled_roc_auc:.4f}")

    if shuffled_roc_auc <= 0.55 and pr_auc_delta <= 0.03:
        print("Sanity check passed: performance collapses after " "target shuffling.")
    elif shuffled_roc_auc <= 0.55:
        print(
            "Sanity check is borderline: ROC-AUC collapsed to chance, "
            "but PR-AUC stayed slightly above prevalence."
        )
    else:
        print("Warning: shuffled-target performance is unexpectedly strong.")


def print_model_comparison_table(results: list[dict[str, Any]]) -> None:
    """Print a ranked comparison table for all evaluated candidates."""
    comparison_df = pd.DataFrame(
        {
            "model_name": [result["model_name"] for result in results],
            "pr_auc": [result["pr_auc"] for result in results],
            "roc_auc": [result["roc_auc"] for result in results],
        }
    )
    comparison_df["rank"] = (
        comparison_df[["pr_auc", "roc_auc"]]
        .apply(tuple, axis=1)
        .rank(method="dense", ascending=False)
        .astype(int)
    )
    comparison_df = comparison_df.sort_values(
        by=["pr_auc", "roc_auc"],
        ascending=False,
    )
    comparison_df["pr_auc"] = comparison_df["pr_auc"].map(lambda x: f"{x:.4f}")
    comparison_df["roc_auc"] = comparison_df["roc_auc"].map(lambda x: f"{x:.4f}")

    print("\nModel comparison table")
    print("----------------------")
    print(comparison_df.to_string(index=False))


def report_top_feature_importance(
    best_result: dict[str, Any],
    top_n: int = 15,
) -> None:
    """Print top feature importances when Random Forest is selected."""
    if best_result["model_name"] != "random_forest":
        print(
            "\nTop feature importance report is skipped because the "
            "selected model is not Random Forest."
        )
        return

    pipeline: Pipeline = best_result["pipeline"]
    preprocessor: ColumnTransformer = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["model"]

    if not hasattr(model, "feature_importances_"):
        print("\nSelected model does not expose feature_importances_.")
        return

    feature_names = preprocessor.get_feature_names_out()
    importances = model.feature_importances_
    feature_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importances,
        }
    ).sort_values(by="importance", ascending=False)

    print("\nTop feature importance (Random Forest)")
    print("--------------------------------------")
    print(feature_df.head(top_n).to_string(index=False))

    hour_related = feature_df[
        feature_df["feature"].str.contains("trans_hour", regex=False)
    ]
    if not hour_related.empty:
        print("\nHour-related feature importance")
        print("------------------------------")
        print(hour_related.to_string(index=False))


def train_and_select_model() -> Pipeline:
    """Main entrypoint: prepare data, compare models, validate, save."""
    # 1) Load and clean rows with invalid labels or duplicate transactions.
    df = load_raw_data()
    x, y, summary = prepare_training_dataframe(df)

    print("\nTraining data summary")
    print("---------------------")
    print(f"Rows before deduplication  : {summary['before_rows']:,}")
    print(f"Rows after deduplication   : {summary['after_rows']:,}")
    print(f"Duplicates removed         : {summary['removed_duplicates']:,}")
    print(f"Invalid-target rows dropped: {summary['invalid_target_rows']:,}")
    print(f"Fraudulent transactions    : {y.sum():,}")
    print(f"Non-fraudulent transactions: {(y == 0).sum():,}")
    print(f"Fraud rate                 : {y.mean():.4%}")

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 2) Sanity checks before training.
    check_for_leakage(x_train, x_test)

    results: list[dict[str, Any]] = []

    print("Model evaluation on holdout set")
    # 3) Train and evaluate each candidate on the same split.
    for (
        model_name,
        (estimator, scale_numeric),
    ) in get_model_candidates().items():
        pipeline = build_model_pipeline(
            x_train,
            estimator=estimator,
            scale_numeric=scale_numeric,
        )
        results.append(
            evaluate_model(
                name=model_name,
                pipeline=pipeline,
                x_train=x_train,
                x_test=x_test,
                y_train=y_train,
                y_test=y_test,
            )
        )

    results.sort(
        key=lambda item: (item["pr_auc"], item["roc_auc"]),
        reverse=True,
    )
    best_result = results[0]

    print("\nModel comparison summary")
    print("------------------------")
    for result in results:
        print(
            f"{result['model_name']}: "
            f"PR-AUC={result['pr_auc']:.4f}, "
            f"ROC-AUC={result['roc_auc']:.4f}"
        )

    print_model_comparison_table(results)

    print(
        "\nSelected model: "
        f"{best_result['model_name']} "
        f"(best PR-AUC, tie-broken by ROC-AUC)"
    )

    # 4) Deeper checks on the selected model only.
    validate_selected_model(
        best_result=best_result,
        x_train=x_train,
        x_test=x_test,
        y_train=y_train,
        y_test=y_test,
    )

    report_top_feature_importance(best_result, top_n=15)

    # 5) Save the full pipeline (feature engineering + preprocessing + model).
    save_model(best_result["pipeline"])

    return best_result["pipeline"]


if __name__ == "__main__":
    train_and_select_model()
