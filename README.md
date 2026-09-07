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