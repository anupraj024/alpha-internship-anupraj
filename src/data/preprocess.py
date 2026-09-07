"""Cleaning and feature/target preparation for credit-risk data."""

import pandas as pd

TARGET_COLUMN = "default_flag"
IDENTIFIER_COLUMNS = ["application_id"]


def clean_loan_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated, cleaned copy without silently changing row meaning."""
    cleaned = data.copy()
    cleaned.columns = cleaned.columns.str.strip()
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)

    numeric_columns = cleaned.select_dtypes(include="number").columns
    if cleaned[numeric_columns].isna().any().any():
        raise ValueError("Numeric columns contain missing values; inspect the source data before training")

    cleaned[TARGET_COLUMN] = pd.to_numeric(cleaned[TARGET_COLUMN], errors="raise").astype(int)
    if not cleaned[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError(f"{TARGET_COLUMN} must contain only 0 and 1")
    return cleaned


def split_features_target(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate target and remove identifiers that cannot generalize to new applications."""
    if TARGET_COLUMN not in data.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' is required")
    features = data.drop(columns=[TARGET_COLUMN, *IDENTIFIER_COLUMNS], errors="ignore")
    target = data[TARGET_COLUMN]
    return features, target
