"""Train and evaluate the Checkpoint 1 credit-risk baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data.load_data import load_loan_data
from src.data.preprocess import clean_loan_data, split_features_target

RANDOM_STATE = 42


def build_model_pipeline(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    """Build a preprocessing and logistic-regression pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric_features),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Return held-out classification metrics, including the Gini coefficient."""
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    roc_auc = roc_auc_score(y_test, probabilities)

    return {
        "roc_auc": float(roc_auc),
        "gini": float(2 * roc_auc - 1),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
    }


def train_and_evaluate(
    data_path: str | Path,
    output_dir: str | Path = "data/processed",
) -> dict:
    """Train the baseline and save metrics and an EDA class-balance plot."""
    data = clean_loan_data(load_loan_data(data_path))
    X, y = split_features_target(data)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    numeric_features = X.select_dtypes(include="number").columns.tolist()
    categorical_features = X.select_dtypes(exclude="number").columns.tolist()
    model = build_model_pipeline(numeric_features, categorical_features)
    model.fit(X_train, y_train)
    metrics = evaluate_model(model, X_test, y_test)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "checkpoint_1_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    ax = data["default_flag"].value_counts().sort_index().plot(
        kind="bar", color=["#2f6f95", "#d46a4c"], title="Default Flag Distribution"
    )
    ax.set_xlabel("Default flag (0 = no default, 1 = default)")
    ax.set_ylabel("Applications")
    figure = ax.get_figure()
    figure.tight_layout()
    figure.savefig(output_path / "default_distribution.png", dpi=150)
    plt.close(figure)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        default="data/raw/finwise_loan_applications.csv",
        help="Path to the raw loan application CSV.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for metrics and generated plots.",
    )
    args = parser.parse_args()
    metrics = train_and_evaluate(args.data_path, args.output_dir)
    print(json.dumps({key: metrics[key] for key in ("roc_auc", "gini", "accuracy", "f1")}, indent=2))


if __name__ == "__main__":
    main()
