---
layout: default
title: Customer Churn Intelligence Dashboard
description: A production-style customer churn prediction, scoring, and reporting system.
---
<div class="content-page project-page" markdown="1">

<div class="project-hero">
  <span class="eyebrow">Production ML case study</span>
  <h1>Retention Risk Workbench</h1>
  <p class="project-lead">An end-to-end churn intelligence system designed to identify at-risk customers, expose model reasoning, and deliver retention insights through a scoring API and business-facing dashboard.</p>
</div>

<div class="button-row project-actions">
  <a class="button" href="https://retention-risk-workbench.streamlit.app/" target="_blank" rel="noopener">Launch live Streamlit app</a>
  <a class="button-secondary" href="https://github.com/liamge/retention-risk-workbench.git" target="_blank" rel="noopener">View GitHub repo</a>
</div>

<div class="proof-strip hero-proof">
  <span>ROC-AUC 0.846</span>
  <span>PR-AUC 0.670</span>
  <span>Recall 0.786 at threshold 0.253</span>
  <span>API + dashboard delivery</span>
</div>

<div class="mini-metric-grid three-up project-facts">
  <article class="mini-metric-card">
    <strong>Role of the system</strong>
    <span>Translate churn risk into actionable retention priorities for business and technical stakeholders.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Primary stack</strong>
    <span>Python, scikit-learn, SHAP, MLflow, FastAPI, Streamlit, Docker, Prometheus, Grafana.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Core product surfaces</strong>
    <span>Training pipeline, scoring service, artifact bundle, dashboard, and deployment scaffolding.</span>
  </article>
</div>

## The Problem

Most churn work stops at a model score or notebook experiment. That leaves several hard questions unanswered: how the score is served, how business users interpret it, which threshold should drive action, how artifact outputs stay synchronized across interfaces, and how the system would be monitored once deployed.

This project was built to bridge that gap. The goal was not just to predict churn, but to package retention risk into a system that can support prioritization, stakeholder communication, and production-style delivery patterns.

## The Solution

<div class="mini-metric-grid two-up">
  <article class="mini-metric-card">
    <strong>Modeling workflow</strong>
    <span>Train and compare candidate models, select a champion, optimize the operating threshold, and export reusable evaluation artifacts.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Serving layer</strong>
    <span>Expose scoring via FastAPI with probability, risk tier, and action guidance so downstream consumers can use a stable interface.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Dashboard layer</strong>
    <span>Surface executive rollups, technical diagnostics, model comparisons, high-risk accounts, and optional uplift-oriented views.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Ops layer</strong>
    <span>Demonstrate Docker, monitoring, MLflow, CI-friendly structure, and deployment scaffolding that make the project feel production-minded.</span>
  </article>
</div>

## System Architecture

<div class="architecture-stack">
  <div class="architecture-step">
    <strong>1. Data and feature layer</strong>
    <p>Start from the IBM Telco churn schema and engineer signals such as price ratio, revenue-risk proxy, service adoption rate, billing behavior, customer lifecycle stage, and contract/payment indicators.</p>
  </div>
  <div class="architecture-step">
    <strong>2. Training and selection</strong>
    <p>Train logistic regression and, when available, XGBoost; compare performance; tune the operating threshold on development data; and persist the champion model with metrics, plots, and metadata.</p>
  </div>
  <div class="architecture-step">
    <strong>3. Artifact bundle</strong>
    <p>Write model files, figures, comparison tables, SHAP summaries, and prediction outputs so the dashboard and API both consume the same source of truth.</p>
  </div>
  <div class="architecture-step">
    <strong>4. Delivery surfaces</strong>
    <p>Serve predictions through FastAPI and present model results in Streamlit, with risk tiers and recommended actions defined consistently across the system.</p>
  </div>
  <div class="architecture-step">
    <strong>5. Deployment patterns</strong>
    <p>Containerize the API and dashboard, wire metrics for Prometheus and Grafana, and include Kubernetes-oriented scaffolding to demonstrate how the service would be operated.</p>
  </div>
</div>

## Modeling Approach

The champion model on the Telco baseline is logistic regression, with XGBoost finishing as a close runner-up within roughly one percentage point of ROC-AUC. That choice is useful for the case study because it shows a disciplined decision process rather than an assumption that the most complex model automatically wins.

Key modeling decisions:

- Compare multiple candidate models rather than hard-coding a single approach.  
- Tune the decision threshold instead of defaulting to 0.5.  
- Save metrics and comparison tables to reusable artifacts.  
- Generate SHAP-based summaries so model behavior can be explained in business-friendly themes.  
- Group feature drivers into interpretable categories such as Billing & Contract, Pricing & Value, Product Usage, Support & Protection, and Customer Lifecycle/Profile.

## What the Dashboard Communicates

<div class="mini-metric-grid two-up">
  <article class="mini-metric-card">
    <strong>Executive summary</strong>
    <span>High-risk counts, revenue-at-risk framing, recommended actions, and grouped churn drivers for stakeholders who need quick directional guidance.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Technical diagnostics</strong>
    <span>ROC, precision-recall, calibration, confusion views, SHAP importance, and metadata that help evaluate model quality and operating behavior.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Model comparison</strong>
    <span>Side-by-side candidate metrics so users can inspect why the champion model was selected.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Risk prioritization</strong>
    <span>High-risk customer views and action guidance that shift the experience from analysis into decision support.</span>
  </article>
</div>

## Why This Feels Production-Style

This repo is strongest not because of any single metric, but because the modeling work is packaged as a usable system.

- The training flow is config-driven and artifact-oriented.  
- The API and dashboard are designed to stay synchronized with the latest exported outputs.  
- The inference layer returns more than a raw score; it returns risk classification and action guidance.  
- MLflow, tests, Docker, and deployment scaffolding show how the project would fit into a more realistic engineering environment.  
- Prometheus and Grafana support an observability story instead of treating deployment as an afterthought.

## What This Project Demonstrates

<div class="proof-strip">
  <span>End-to-end delivery</span>
  <span>Threshold-aware classification</span>
  <span>Explainability for stakeholders</span>
  <span>Monitorable inference service</span>
  <span>Dashboard-driven decision support</span>
</div>

For a hiring manager, client, or collaborator, the point of this project is that it shows how I think across the whole lifecycle of a predictive system: problem framing, feature engineering, evaluation, interpretation, product surfaces, and operational readiness.

## Extension Paths

The repo also leaves room for more advanced decision intelligence extensions:

- blend churn probability with contract value to estimate ARR or revenue at risk  
- add calibrated or gradient-boosted alternatives and re-run threshold selection  
- connect the dashboard directly to live API scoring rather than batch prediction files  
- expand the uplift and policy layer for retention targeting under budget constraints  
- add drift checks and promotion logic for a fuller model lifecycle story

</div>
