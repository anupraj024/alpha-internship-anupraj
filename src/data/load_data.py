"""Data loading helpers for the credit-risk project."""

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {
    "application_id",
    "age",
    "income_annual_inr",
    "employment_type",
    "employment_years",
    "existing_loans",
    "existing_emi_inr",
    "credit_score",
    "loan_amount_inr",
    "loan_purpose",
    "ltv_ratio",
    "dti_ratio",
    "default_flag",
}


def load_loan_data(file_path: str | Path) -> pd.DataFrame:
    """Load the raw CSV and validate the expected schema."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Loan data file not found: {path}")

    data = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")
    return data
