# Explainable Credit Risk Scoring System

## Overview

This project develops an explainable machine learning system for
credit risk scoring in the FinTech domain.

The system will use loan application information such as income,
employment history, credit history, and loan amount to estimate
credit risk.

The final system will combine:

- XGBoost for credit risk prediction
- SHAP for model explainability
- FastAPI for REST API serving
- React for the user interface
- MLflow for experiment tracking

## Problem Statement

Traditional credit scoring systems can provide a prediction without
clearly explaining the factors behind that prediction.

The objective of this project is to build a credit risk scoring
system that provides both:

1. A risk prediction/score
2. An explanation of the most important factors influencing the prediction

## Objectives

- Build a reliable credit risk classification model
- Achieve a Gini coefficient of at least 0.45 on the test set
- Provide per-application explanations using SHAP
- Categorize applicants into low, medium, and high risk
- Provide predictions through a REST API
- Build a React-based frontend
- Test API performance under 50 concurrent requests
- Conduct a bias audit across gender and age groups

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Model | XGBoost |
| Explainability | SHAP |
| API | FastAPI |
| Frontend | React |
| Experiment Tracking | MLflow |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Testing | Pytest |

## Project Architecture

```text
React Frontend
      |
      v
FastAPI REST API
      |
      v
Preprocessing Pipeline
      |
      v
XGBoost Model
      |
      +----> Risk Score
      |
      +----> SHAP Explanation
      |
      v
Prediction + Top Risk Factors

## Checkpoint 1: Baseline Credit-Risk Model

Checkpoint 1 establishes a reproducible tabular ML baseline before adding
explainability, an API, or a frontend. It includes:

- CSV loading with schema validation
- Duplicate and target validation
- Exploratory analysis of class balance and numeric relationships
- Leakage-safe feature preparation
- A stratified 80/20 train/test split with `random_state=42`
- A balanced logistic-regression baseline with one-hot encoding and scaling
- ROC AUC, Gini, accuracy, precision, recall, F1, and a confusion matrix

The `application_id` column is intentionally excluded from model features
because it is an identifier rather than a generalizable applicant attribute.
All learned transformations are fitted within the scikit-learn pipeline using
the training split only.

### Setup

From the repository root, create or activate a virtual environment and install
the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Run Checkpoint 1

Train and evaluate the baseline on the included raw data:

```powershell
python -m src.models.train_model
```

The command writes `data/processed/checkpoint_1_metrics.json` and
`data/processed/default_distribution.png`. It prints ROC AUC, Gini, accuracy,
and F1 for the held-out test set. The dataset is included for development;
performance targets should be confirmed against an approved validation set
before making production claims.

Run the focused tests with:

```powershell
python -m pytest tests/test_checkpoint_1.py -q
```

Open `notebooks/checkpoint_1.ipynb` for the walkthrough covering objective,
data inspection, cleaning, EDA, feature preparation, model preparation,
evaluation, and conclusion.