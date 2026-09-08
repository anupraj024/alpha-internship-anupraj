# Explainable Credit Risk Scoring System

> A production-grade, explainable machine learning system for credit risk
> scoring in the FinTech domain — combining XGBoost, SHAP, FastAPI, and React.

---

## Table of Contents

1. [Overview](#overview)
2. [Literature Review](#literature-review)
3. [Model Architecture](#model-architecture)
4. [Evaluation Metrics](#evaluation-metrics)
5. [Data Pipeline Skeleton](#data-pipeline-skeleton)
6. [Project Structure](#project-structure)
7. [Technology Stack](#technology-stack)
8. [Setup & Quickstart](#setup--quickstart)
9. [Checkpoints](#checkpoints)

---

## Overview

This project develops an explainable machine learning system for credit risk
scoring in the FinTech domain.

The system uses loan application attributes — income, employment history,
credit history, loan amount, and debt metrics — to estimate the probability
of default and categorise applicants into **Low / Medium / High** risk bands.

The full system combines:

- **XGBoost** for tabular credit risk prediction
- **SHAP (TreeSHAP)** for per-application model explainability
- **FastAPI** for REST API serving
- **React** for the user-facing dashboard
- **MLflow** for experiment tracking and model registry

### Problem Statement

Traditional credit scoring systems produce a decision without transparently
explaining the factors behind it. This project builds a system that provides
**both** a calibrated risk score *and* a ranked list of the features that drove
that score — satisfying regulatory requirements (e.g., EU AI Act, ECOA) and
building loan-officer trust.

### Objectives

- Build a reliable credit-risk binary classification model
- Achieve a **Gini coefficient >= 0.45** on the held-out test set
- Provide per-application SHAP-based explanations with top-N risk factors
- Categorise applicants into Low / Medium / High risk tiers
- Serve predictions through a REST API (p95 latency < 200 ms)
- Build a React frontend that renders risk scores and SHAP waterfall charts
- Pass load testing at 50 concurrent requests
- Conduct a bias audit across gender and age groups

---

## Literature Review

The three references below directly shaped the design decisions in this project.

### 1. Lundberg & Lee (2017) — *A Unified Approach to Interpreting Model Predictions*

> **Paper:** Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach to
> Interpreting Model Predictions. *Advances in Neural Information Processing
> Systems (NeurIPS)*, 30.
> arXiv:1705.07874 — https://arxiv.org/abs/1705.07874

**What it contributes:**
Introduces **SHAP (SHapley Additive exPlanations)**, a game-theory framework
that assigns each feature a consistent, additive contribution to a model's
output. TreeSHAP — the fast variant for tree-based models — runs in
polynomial time and is the explainability backbone of this project.

**Design decisions taken from this work:**

| Insight | Applied Here |
|---|---|
| Shapley values are the unique solution satisfying efficiency, symmetry, and consistency | Use `shap.TreeExplainer` over LIME or permutation importance |
| Global importance = mean(|SHAP|) across the dataset | Report both local (per-application) and global feature rankings |
| SHAP interaction values capture non-linear feature pairs | Planned for advanced explainability checkpoint |

---

### 2. Kozodoi (2021) — *Home Credit Default Risk (Top-4% Kaggle Solution)*

> **Repository:** https://github.com/kozodoi/Kaggle_Home_Credit
> **Competition:** Home Credit Default Risk — Kaggle (ROC-AUC leaderboard task)

**What it contributes:**
A rigorously structured end-to-end pipeline for credit default prediction.
Demonstrates how feature aggregation from relational tables, stratified
K-fold cross-validation, and LightGBM/XGBoost ensembling together push
ROC-AUC beyond 0.80 on a real-world imbalanced dataset.

**Design decisions taken from this work:**

| Insight | Applied Here |
|---|---|
| Aggregate relational tables (mean/max/min) before joining to the application table | Planned for multi-source data ingestion module |
| Stratified K-fold CV prevents leakage from class imbalance | Implemented in `train_model.py` with `StratifiedKFold` |
| Feature selection via GBDT importance reduces noise | Use `feature_importances_` post-training to prune low-signal columns |
| Bayesian hyperparameter tuning (Optuna) outperforms grid search | Planned for Checkpoint 3 tuning stage |

---

### 3. Armanpanigrahi59 (2024) — *Credit Risk Scoring Engine*

> **Repository:** https://github.com/armanpanigrahi59/Credit-Risk-Scoring-Engine

**What it contributes:**
A production-ready reference architecture that bundles an **XGBoost model**
with a **FastAPI backend**, a **React-Vite frontend**, and real-time SHAP
inference — achieving p95 API latency < 200 ms. Provides concrete Pydantic
schemas, SHAP waterfall rendering, and a Docker compose setup for deployment.

**Design decisions taken from this work:**

| Insight | Applied Here |
|---|---|
| Pydantic `ApplicationInput` schema for strict API validation | Adopted in `src/api/schemas.py` |
| Return SHAP values as a JSON array alongside the risk score | `/predict` endpoint returns `{ risk_score, risk_tier, shap_values }` |
| React waterfall chart from the `shap` JS library | Planned for frontend checkpoint |
| Docker multi-stage build keeps the image under 500 MB | Dockerfile planned for deployment checkpoint |

---

## Model Architecture

```text
+-------------------------------------------------------------+
|                     Data Ingestion Layer                     |
|   CSV / API JSON  -->  Schema Validation  -->  DataFrame     |
+----------------------------+--------------------------------+
                             |
+----------------------------v--------------------------------+
|                   Preprocessing Pipeline                     |
|  +--------------+  +------------------+  +---------------+  |
|  |  Imputation  |  |  OneHotEncoding  |  | StandardScaler|  |
|  |  (Median /   |  |  (employment_    |  | (numeric      |  |
|  |   Mode)      |  |   type, purpose) |  |  features)    |  |
|  +--------------+  +------------------+  +---------------+  |
|              Fitted on TRAINING split only                    |
+----------------------------+--------------------------------+
                             |
+----------------------------v--------------------------------+
|                       Model Layer                            |
|                                                              |
|   Stage 1 - Baseline                                         |
|   +--------------------------------------------------+       |
|   |  Logistic Regression  (class_weight=balanced)    |       |
|   |  Establishes minimum Gini / AUC bar              |       |
|   +--------------------------------------------------+       |
|                             |                                |
|   Stage 2 - Primary Model                                    |
|   +--------------------------------------------------+       |
|   |  XGBoost Classifier                              |       |
|   |    objective        : binary:logistic            |       |
|   |    eval_metric      : auc                        |       |
|   |    scale_pos_weight : (neg / pos count)          |       |
|   |    n_estimators     : 300  (early stopping)      |       |
|   |    max_depth        : 6                          |       |
|   |    learning_rate    : 0.05                       |       |
|   |    subsample / colsample_bytree : 0.8            |       |
|   +--------------------------------------------------+       |
|                             |                                |
|   Stage 3 - Hyperparameter Tuning  (Checkpoint 3)           |
|   +--------------------------------------------------+       |
|   |  Optuna - 100 trials, StratifiedKFold (k=5)     |       |
|   |  Optimise: ROC-AUC on OOF predictions           |       |
|   +--------------------------------------------------+       |
+----------------------------+--------------------------------+
                             |
+----------------------------v--------------------------------+
|                   Explainability Layer                       |
|                                                              |
|   shap.TreeExplainer(xgb_model)                             |
|     +-- shap_values  [n_samples x n_features]               |
|     +-- Global importance  = mean(|shap|) per feature       |
|     +-- Local explanation = top-5 features per applicant    |
+----------------------------+--------------------------------+
                             |
+----------------------------v--------------------------------+
|                   Risk Tier Mapping                          |
|                                                              |
|   P(default) >= 0.65          -->  HIGH   risk (reject)     |
|   P(default) in [0.35, 0.65)  -->  MEDIUM risk (review)     |
|   P(default) < 0.35           -->  LOW    risk (approve)     |
+---------------+-----------------------------+---------------+
                |                             |
                v                             v
        FastAPI REST API               MLflow Registry
        /predict  -->  JSON            Logged runs, params,
        { risk_score,                  metrics, artefacts
          risk_tier,
          shap_values,
          top_factors }
                |
                v
          React Dashboard
          Risk gauge + SHAP waterfall
```

### Feature Set (v1)

| Feature | Type | Description |
|---|---|---|
| `age` | Numeric | Applicant age in years |
| `income_annual_inr` | Numeric | Annual income (INR) |
| `employment_type` | Categorical | Salaried / Self-employed / Unemployed |
| `employment_years` | Numeric | Years at current employer |
| `existing_loans` | Numeric | Count of open loan accounts |
| `existing_emi_inr` | Numeric | Monthly EMI obligation (INR) |
| `credit_score` | Numeric | Bureau credit score |
| `loan_amount_inr` | Numeric | Requested loan amount (INR) |
| `loan_purpose` | Categorical | Home / Car / Education / Personal / Business |
| `ltv_ratio` | Numeric | Loan-to-Value ratio (%) |
| `dti_ratio` | Numeric | Debt-to-Income ratio (%) |

`application_id` is always excluded — it is an identifier, not a
generalizable applicant attribute.

---

## Evaluation Metrics

The following metrics are computed on the **held-out test split** (stratified
80 / 20, `random_state=42`). All transformations are fit on training data only.

### Primary Metrics

| Metric | Formula | Target | Rationale |
|---|---|---|---|
| **ROC-AUC** | Area under ROC curve | >= 0.75 | Threshold-independent; standard for imbalanced classification |
| **Gini Coefficient** | `2 x AUC - 1` | >= 0.45 | Industry convention in credit scoring; directly comparable across models |
| **KS Statistic** | `max(CDF_good - CDF_bad)` | >= 0.35 | Measures maximum separation between defaulters and non-defaulters |

### Secondary Metrics

| Metric | Description |
|---|---|
| **Precision** | Of all predicted defaults, fraction that are true defaults |
| **Recall** | Of all actual defaults, fraction correctly flagged |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Accuracy** | Overall classification accuracy (reported but not optimised) |
| **Confusion Matrix** | TP / FP / FN / TN counts at the 0.5 probability threshold |
| **Brier Score** | Mean squared error of calibrated probabilities |

### Fairness / Bias Metrics *(Checkpoint 5)*

| Metric | Description |
|---|---|
| **Demographic Parity** | Difference in approval rates across gender / age groups |
| **Equal Opportunity** | Difference in True Positive Rates across groups |
| **PSI** | Population Stability Index — detects distribution drift over time |

### Metric Interpretation Guide

```text
Gini  < 0.30         Weak model; revisit features or data quality
Gini  0.30 - 0.45    Acceptable baseline
Gini  >= 0.45        TARGET — production-ready discrimination

KS    < 0.20         Poor separation; model cannot reliably set a cut-off
KS    0.20 - 0.40    Acceptable separation
KS    >= 0.40        Strong separation
```

---

## Data Pipeline Skeleton

```text
data/
+-- raw/                         # Source CSVs (never modified in-place)
|   +-- loan_applications.csv
+-- processed/                   # Generated artefacts
    +-- checkpoint_1_metrics.json
    +-- default_distribution.png
    +-- (future checkpoints...)

src/data/
+-- __init__.py
+-- load_data.py                 # Schema validation, CSV --> DataFrame
+-- preprocess.py                # Cleaning, feature/target split

Pipeline execution order:
  1. load_data.load_loan_data(path)
       Validates all 13 required columns are present
       Returns raw pd.DataFrame — no transformation applied

  2. preprocess.clean_loan_data(df)
       Strips whitespace from column names
       Drops exact duplicate rows
       Raises ValueError if any numeric column contains NaN
       Casts default_flag to int and validates it is strictly {0, 1}

  3. preprocess.split_features_target(df)
       Drops application_id (identifier, not a feature)
       Returns (X: pd.DataFrame, y: pd.Series)

  4. sklearn.Pipeline([
         OneHotEncoder(handle_unknown="ignore"),
         StandardScaler(),
         XGBoostClassifier(...)
     ]).fit(X_train, y_train)
       All transforms fitted on training split ONLY

  5. model.predict_proba(X_test)[:, 1]
       Produces default probability for each test applicant

  6. shap.TreeExplainer(model).shap_values(X_test)
       Generates local SHAP contributions for each applicant
       Global importance = mean(|shap|) across test set
```

### Required CSV Columns

The source file must contain these 13 columns (order does not matter):

```
application_id, age, income_annual_inr, employment_type,
employment_years, existing_loans, existing_emi_inr, credit_score,
loan_amount_inr, loan_purpose, ltv_ratio, dti_ratio, default_flag
```

---

## Project Structure

```text
credit-risk-scoring/
+-- data/
|   +-- raw/                     # Source data (git-ignored)
|   +-- processed/               # Generated metrics & plots
+-- docs/                        # Additional documentation
+-- frontend/                    # React + Vite dashboard (future)
+-- notebooks/
|   +-- checkpoint_1.ipynb       # EDA -> baseline model walkthrough
+-- src/
|   +-- __init__.py
|   +-- api/                     # FastAPI app (future checkpoint)
|   +-- data/
|   |   +-- load_data.py         # CSV ingestion + schema validation
|   |   +-- preprocess.py        # Cleaning + feature/target split
|   +-- explainability/          # SHAP wrappers (future checkpoint)
|   +-- models/
|       +-- train_model.py       # Sklearn pipeline + XGBoost + metrics
+-- tests/
|   +-- test_checkpoint_1.py     # Pytest suite for Checkpoint 1
+-- conftest.py                  # Shared pytest fixtures
+-- requirements.txt
+-- README.md
```

---

## Technology Stack

| Component | Technology | Version |
|---|---|---|
| Language | Python | >= 3.10 |
| ML Model | XGBoost | >= 2.0 |
| Explainability | SHAP (TreeSHAP) | >= 0.43 |
| ML Framework | scikit-learn | >= 1.2 |
| Data Processing | Pandas, NumPy | >= 2.0 / >= 1.24 |
| API | FastAPI + Uvicorn | >= 0.110 |
| Frontend | React + Vite | — |
| Experiment Tracking | MLflow | >= 2.12 |
| Visualisation | Matplotlib, Seaborn | >= 3.7 / >= 0.12 |
| Testing | Pytest | >= 7.0 |
| Notebooks | Jupyter / ipykernel | >= 1.0 / >= 6.0 |

---

## Setup & Quickstart

### Prerequisites

- Python 3.10 or higher
- Git

### 1. Clone & create environment

```powershell
git clone <repo-url>
cd credit-risk-scoring
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### 2. Place raw data

```
data/raw/loan_applications.csv
```

---

## Checkpoints

### Checkpoint 1 — Baseline Credit-Risk Model (DONE)

Establishes a reproducible tabular ML baseline before adding explainability,
an API, or a frontend. It includes:

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

**Run it:**

```powershell
python -m src.models.train_model
```

Outputs `data/processed/checkpoint_1_metrics.json` and
`data/processed/default_distribution.png`.

**Tests:**

```powershell
python -m pytest tests/test_checkpoint_1.py -q
```

**Notebook:** `notebooks/checkpoint_1.ipynb` covers objective, data
inspection, cleaning, EDA, feature preparation, model training, evaluation,
and conclusion.

---

### Checkpoint 2 — XGBoost + SHAP Explainability (PLANNED)

- Replace logistic regression with XGBoost (`scale_pos_weight`, early stopping)
- Integrate `shap.TreeExplainer` for global and local explanations
- Add KS Statistic and Brier Score to metric suite
- Log runs and artefacts to MLflow

### Checkpoint 3 — Hyperparameter Tuning with Optuna (PLANNED)

- 100-trial Optuna study, StratifiedKFold (k=5)
- Optimise ROC-AUC on out-of-fold predictions
- Compare tuned XGBoost vs. baseline Gini

### Checkpoint 4 — FastAPI Serving (PLANNED)

- `/predict` endpoint: accepts `ApplicationInput` JSON, returns risk score,
  risk tier, and top-5 SHAP feature contributions
- Pydantic validation, health-check route
- Load test at 50 concurrent requests

### Checkpoint 5 — Bias Audit & React Dashboard (PLANNED)

- Demographic parity and equal opportunity across gender / age cohorts
- PSI for population drift monitoring
- React + Vite dashboard: risk gauge, SHAP waterfall chart, feature table