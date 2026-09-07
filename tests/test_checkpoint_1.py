from pathlib import Path

import pandas as pd

from src.data.load_data import load_loan_data
from src.data.preprocess import clean_loan_data, split_features_target
from src.models.train_model import train_and_evaluate

DATA_PATH = Path("data/raw/finwise_loan_applications.csv")


def test_load_and_clean_data() -> None:
    data = clean_loan_data(load_loan_data(DATA_PATH))

    assert len(data) == 1000
    assert data.isna().sum().sum() == 0
    assert set(data["default_flag"].unique()) == {0, 1}


def test_application_id_is_not_a_model_feature() -> None:
    data = clean_loan_data(load_loan_data(DATA_PATH))
    features, target = split_features_target(data)

    assert "application_id" not in features.columns
    assert target.name == "default_flag"
    assert isinstance(features, pd.DataFrame)


def test_training_produces_held_out_metrics(tmp_path: Path) -> None:
    metrics = train_and_evaluate(DATA_PATH, tmp_path)

    assert 0 <= metrics["roc_auc"] <= 1
    assert -1 <= metrics["gini"] <= 1
    assert (tmp_path / "checkpoint_1_metrics.json").exists()
    assert (tmp_path / "default_distribution.png").exists()
