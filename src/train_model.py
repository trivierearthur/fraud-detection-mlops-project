from pathlib import Path
import sys

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent

try:
    from src.preprocess import preprocess_dataframe
    from src.data_loader import load_raw_data
except ModuleNotFoundError:
    try:
        from fraud_detection_mlops_project.src.preprocess import (
            preprocess_dataframe,
        )
        from fraud_detection_mlops_project.src.data_loader import load_raw_data
    except ModuleNotFoundError:
        if str(CURRENT_DIR) not in sys.path:
            sys.path.append(str(CURRENT_DIR))
        from preprocess import preprocess_dataframe
        from data_loader import load_raw_data


def build_model_pipeline(x: pd.DataFrame) -> Pipeline:
    numeric_features = x.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = x.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
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

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


def train_random_forest() -> None:
    df = load_raw_data()
    x, y = preprocess_dataframe(df)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    pipeline = build_model_pipeline(x_train)
    pipeline.fit(x_train, y_train)

    y_pred = pipeline.predict(x_test)
    y_score = pipeline.predict_proba(x_test)[:, 1]

    print("Model evaluation on holdout set")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_score):.4f}")
    print(f"PR-AUC: {average_precision_score(y_test, y_score):.4f}")


if __name__ == "__main__":
    train_random_forest()
