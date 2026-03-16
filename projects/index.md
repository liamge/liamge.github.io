---
layout: default
title: Projects
description: Selected machine learning, NLP, and decision intelligence projects by Liam Geron.
---
<div class="content-page project-page" markdown="1">

<div class="project-hero">
  <span class="eyebrow">Selected work</span>
  <h1>Case studies built around business decisions, not just models.</h1>
  <p class="project-lead">These projects are designed to show how I approach end-to-end delivery: problem framing, technical design, evaluation, product surfaces, and the operational details that make systems usable by real stakeholders.</p>
</div>

<div class="featured-project-block">
  <div>
    <span class="eyebrow">Featured case study</span>
    <h2>Retention Risk Workbench</h2>
    <p>A production-style churn intelligence system that moves from feature engineering and model selection to API delivery, dashboard reporting, explainability, and monitoring-oriented deployment patterns.</p>
    <div class="tag-row">
      <span class="tag">Python</span>
      <span class="tag">scikit-learn</span>
      <span class="tag">FastAPI</span>
      <span class="tag">Streamlit</span>
      <span class="tag">MLflow</span>
    </div>
    <div class="proof-strip">
      <span>ROC-AUC 0.846</span>
      <span>Threshold tuning</span>
      <span>API + dashboard</span>
      <span>Monitoring-ready ops layer</span>
    </div>
    <p><a class="button" href="/projects/churn-dashboard">Read the case study</a></p>
  </div>
  <div class="mini-metric-grid">
    <article class="mini-metric-card">
      <strong>Primary problem</strong>
      <span>Which customers are most likely to churn, and how should a retention team act on that risk?</span>
    </article>
    <article class="mini-metric-card">
      <strong>What was built</strong>
      <span>Training pipeline, champion model workflow, scoring API, dashboard, and explainability artifacts.</span>
    </article>
    <article class="mini-metric-card">
      <strong>Why it matters</strong>
      <span>Turns model output into prioritized action, not just a static notebook metric.</span>
    </article>
  </div>
</div>

## Additional Case Studies

<div class="project-card-grid">
  <article class="project-showcase-card">
    <span class="card-kicker">Production ML</span>
    <h3>Customer Churn Intelligence Dashboard</h3>
    <p>End-to-end retention modeling system with artifact-driven training, champion-model selection, SHAP-based interpretation, FastAPI scoring, and a business-facing Streamlit dashboard.</p>
    <div class="proof-strip compact">
      <span>Logistic regression champion</span>
      <span>SHAP driver themes</span>
      <span>Prometheus + Grafana scaffolding</span>
    </div>
    <div class="tag-row">
      <span class="tag">MLflow</span>
      <span class="tag">Docker</span>
      <span class="tag">FastAPI</span>
      <span class="tag">Streamlit</span>
    </div>
    <p><a href="/projects/churn-dashboard">View case study</a></p>
  </article>

  <article class="project-showcase-card">
    <span class="card-kicker">LLM Systems</span>
    <h3>RAG Evaluation Workbench</h3>
    <p>Evaluation-oriented retrieval system focused on measuring answer quality, groundedness, retrieval behavior, and the tradeoffs between chunking, ranking, and prompting choices.</p>
    <div class="proof-strip compact">
      <span>RAG evaluation framing</span>
      <span>Groundedness focus</span>
      <span>Monitoring mindset</span>
    </div>
    <div class="tag-row">
      <span class="tag">LLMs</span>
      <span class="tag">Retrieval</span>
      <span class="tag">Evaluation</span>
      <span class="tag">Monitoring</span>
    </div>
    <p><a href="/projects/rag-eval">View case study</a></p>
  </article>
</div>

## What these case studies are meant to show

<div class="mini-metric-grid three-up">
  <article class="mini-metric-card">
    <strong>Clear problem framing</strong>
    <span>I aim to show the decision the system supports, not just the modeling technique used.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Operational design</strong>
    <span>Training, serving, reporting, and monitoring are treated as part of the same product surface.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Business translation</strong>
    <span>Each project is structured to communicate both technical rigor and practical stakeholder value.</span>
  </article>
</div>

</div>
