---
layout: default
title: Causal Uplift Modeling + ROI Dashboard
description: A decision-focused causal ML case study for uplift estimation, budget-aware targeting, and ROI optimization.
---
<div class="content-page project-page" markdown="1">

<div class="project-hero">
  <span class="eyebrow">Decision intelligence case study</span>
  <h1>Causal Uplift Modeling + ROI Dashboard</h1>
  <p class="project-lead">An end-to-end causal ML system that estimates which customers are most likely to respond to treatment, then converts those predictions into budget-aware targeting recommendations through an interactive ROI dashboard.</p>
</div>

<div class="button-row project-actions">
  <a class="button" href="https://github.com/liamge/causal-uplift-modeling" target="_blank" rel="noopener">View GitHub repo</a>
  <a class="button-secondary" href="https://causal-uplift-modeling.streamlit.app/" target="_blank" rel="noopener">Open live Streamlit app</a>
</div>

<div class="proof-strip hero-proof">
  <span>T-Learner uplift modeling</span>
  <span>Budget-aware policy optimization</span>
  <span>Qini + decile evaluation</span>
  <span>Interactive Streamlit delivery</span>
</div>

<div class="mini-metric-grid three-up project-facts">
  <article class="mini-metric-card">
    <strong>Role of the system</strong>
    <span>Turn treatment-effect estimates into practical targeting decisions instead of stopping at raw conversion propensity.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Primary stack</strong>
    <span>Python, scikit-learn, pandas, Plotly, Streamlit, YAML configuration, and lightweight testing.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Core product surfaces</strong>
    <span>Simulation pipeline, uplift training workflow, ROI policy engine, evaluation artifacts, and business-facing dashboard.</span>
  </article>
</div>

## The Problem

Many marketing and retention models answer the wrong question. They predict who is likely to convert, but they do not estimate who will convert **because of intervention**. That distinction matters when budget is limited and treatment has a real cost.

Without uplift modeling, teams often waste spend on customers who would have converted anyway, miss persuadable customers with genuine incremental upside, and struggle to connect model outputs to campaign economics.

## The Solution

This project reframes the problem around **incremental impact**.

<div class="mini-metric-grid two-up">
  <article class="mini-metric-card">
    <strong>Uplift modeling workflow</strong>
    <span>Train separate treated and control models, estimate individual treatment effect, and rank customers by expected incremental response.</span>
  </article>
  <article class="mini-metric-card">
    <strong>ROI decision layer</strong>
    <span>Translate uplift into expected value, campaign cost, and net value so targeting decisions are grounded in business tradeoffs.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Evaluation layer</strong>
    <span>Measure model usefulness with Qini-style performance curves, uplift-by-decile summaries, and targeting-depth diagnostics.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Delivery layer</strong>
    <span>Expose the results through a Streamlit dashboard where stakeholders can adjust assumptions and explore recommended cutoffs live.</span>
  </article>
</div>

## System Architecture

<div class="architecture-stack">
  <div class="architecture-step">
    <strong>1. Campaign simulation</strong>
    <p>Generate synthetic treatment-control data with configurable treatment rate, response signal, and feature structure so the full workflow can be demonstrated end to end.</p>
  </div>
  <div class="architecture-step">
    <strong>2. Twin-model uplift training</strong>
    <p>Fit separate random forest models for treated and untreated outcomes, then estimate uplift as the difference between predicted response under treatment and control.</p>
  </div>
  <div class="architecture-step">
    <strong>3. Scoring and artifact generation</strong>
    <p>Write scored customer outputs, metrics, Qini curves, decile summaries, and preview tables that can be reused across analysis and dashboard surfaces.</p>
  </div>
  <div class="architecture-step">
    <strong>4. Policy optimization</strong>
    <p>Simulate campaign cost and expected incremental value across targeting depth, then identify the customer cutoff that maximizes net value under a given budget.</p>
  </div>
  <div class="architecture-step">
    <strong>5. Interactive decision support</strong>
    <p>Present the outputs in a dashboard with live controls for conversion value, treatment cost, and campaign budget so non-technical users can test scenarios directly.</p>
  </div>
</div>

## Modeling Approach

The core modeling idea is simple but powerful: estimate two probabilities for each customer and compare them.

- **P(conversion | treated)** estimates expected response if the customer is contacted or offered treatment.  
- **P(conversion | control)** estimates expected response without intervention.  
- **Predicted uplift** is the difference between those two values.

This project uses a **T-Learner** structure with paired Random Forest classifiers. That keeps the implementation approachable while still demonstrating real treatment-effect estimation logic. It also makes the project more useful as a portfolio piece because the causal framing is explicit and directly tied to action.

## What the Dashboard Communicates

<div class="mini-metric-grid two-up">
  <article class="mini-metric-card">
    <strong>Policy curve</strong>
    <span>Shows cumulative expected net value as targeting expands, making the recommended stopping point easy to explain.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Qini curve</strong>
    <span>Helps validate whether the model is separating high-uplift customers from the broader population.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Decile analysis</strong>
    <span>Compares predicted and observed uplift across ranked groups to test ranking stability and calibration in a practical business format.</span>
  </article>
  <article class="mini-metric-card">
    <strong>Top-target preview</strong>
    <span>Lets users inspect the highest-value customers and understand why a particular campaign scope is being recommended.</span>
  </article>
</div>


## What This Project Demonstrates

<div class="proof-strip">
  <span>Causal reasoning in ML</span>
  <span>Decision-focused analytics</span>
  <span>Business-aware targeting strategy</span>
  <span>Interactive stakeholder delivery</span>
  <span>Reusable evaluation artifacts</span>
</div>

For a hiring manager or consulting client, the point of this project is that it shows how I connect ML outputs to operational decisions. Instead of optimizing purely for classification performance, the system is structured around intervention design, campaign economics, and prioritization under constraints.

## Extension Paths

This repo also creates room for deeper decision-intelligence work:

- replace the T-Learner with X-Learner, DR-Learner, or causal forest variants  
- incorporate treatment constraints by channel, segment, or campaign type  
- connect uplift recommendations to revenue tiers or customer lifetime value  
- deploy a public API layer alongside the dashboard for scored-policy delivery  
- swap the synthetic data generator for a real marketing or retention campaign dataset

</div>
