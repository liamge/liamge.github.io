---
layout: default
title: Customer Churn Intelligence Dashboard
description: A production-style customer churn prediction and reporting system.
---
<div class="content-page" markdown="1">

# Customer Churn Intelligence Dashboard

Retention Risk Workbench: a production-style churn prediction system that ships a scoring API, evaluation artifacts, and an interactive dashboard for business and technical stakeholders.

## Snapshot

- Champion model: logistic regression (Telco churn baseline)  
- Test set: ROC-AUC 0.846, PR-AUC 0.670, F1 0.620, recall 0.786 at threshold 0.253 (positive rate 0.409)  
- Candidates: XGBoost runner-up within 1% ROC-AUC  
- Stack: Python, scikit-learn, SHAP, MLflow, FastAPI, Streamlit  
- Assets: metrics, SHAP tables, comparison charts, latest predictions, causal uplift tables

## Product Story

Many churn efforts end at a notebook. This workbench carries the model through feature engineering, model selection, explainability, API packaging, and a dashboard that surfaces both revenue risk and model quality. It is built to be refreshed after every training run so the dashboard and API stay in sync with the latest artifacts.

## Dashboard Views

- Executive Summary: high-risk counts, revenue-at-risk roll-up, recommended actions, driver themes grouped from SHAP.  
- Technical Performance: ROC/PR/calibration/confusion plots, SHAP feature importance, full model metadata.  
- Model Comparison: side-by-side metrics for logistic regression vs XGBoost with metric selector.  
- High-Risk Customers: top 25 customers with probabilities, driver themes, and action guidance.  
- Causal Uplift: optional uplift curve, Qini, and budget-aware policy recommendations.

## Data & Features

Built on the IBM Telco churn schema plus engineered signals such as price ratio, revenue risk proxy, service adoption rate, contract/payment flags (auto-pay, electronic check, paperless), tenure groups, streaming/support usage, and customer lifecycle stages. Feature themes are mapped to business-friendly buckets (Billing & Contract, Pricing & Value, Product Usage, Support & Protection, Customer Lifecycle/Profile) for storytelling in the dashboard.

## Modeling Approach

- Trains logistic regression and (when available) XGBoost; tunes threshold on dev F1 and selects a champion.  
- Stores metrics and comparison table in `artifacts/metrics.json` and `artifacts/candidate_model_results.csv`.  
- Generates SHAP global importance and evaluation figures for dashboard ingestion.  
- Risk tiers and recommended actions are defined in the FastAPI service (`api/main.py`) so scoring and UI stay consistent.

## Delivery & Ops

- Streamlit UI: `dashboard/app.py` reads the latest artifacts and `data/predictions/*.csv`.  
- Scoring API: FastAPI in `api/main.py`, returning probability, risk tier, and action guidance.  
- Training pipeline: `src/train.py` with config-driven splits and preprocessing; outputs champion model to `artifacts/model.pkl`.  
- Causal uplift (optional): `artifacts/causal/` holds uplift tables and budget-aware policy recommendations.

## Run It Locally

```bash
cd retention-risk-workbench
pip install -r requirements.txt
python src/train.py --config configs/base.yaml   # train + export artifacts
streamlit run dashboard/app.py                   # open the dashboard
```

## What to Extend Next

- Add ARR-at-risk calculation by blending churn probability with contract value.  
- Swap in a gradient-boosted or calibrated model and re-run threshold tuning.  
- Connect the FastAPI scoring endpoint to the dashboard for live inference instead of batch CSVs.

</div>
