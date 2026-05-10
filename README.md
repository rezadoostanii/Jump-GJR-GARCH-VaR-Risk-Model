# 🚀 Jump + GJR-GARCH Risk Modeling & VaR Backtesting Engine

## 📌 Overview

This project builds a **quantitative risk engine inspired by institutional risk management systems**.

It combines volatility modeling, jump processes, and statistical backtesting to analyze and forecast financial risk under realistic market conditions.

The main objective is to answer:

> How does an asset behave under normal and extreme market stress conditions?

---

## ⚙️ Key Features

- 📊 GJR-GARCH(1,1) volatility modeling with leverage effect
- ⚡ Jump-diffusion process for modeling extreme market events
- 📉 Value-at-Risk (VaR) and Conditional VaR (CVaR) estimation
- 🔍 Full statistical backtesting framework
- 📈 Rolling window forecasting evaluation
- 🧪 Exception clustering detection
- 💾 SQLite-based financial data integration
- 📊 Monte Carlo simulation and visualization tools

---

## 🧠 Models Used

### 📊 GJR-GARCH(1,1)

Used to model time-varying volatility with asymmetry:

- Negative shocks increase volatility more than positive ones
- Captures volatility clustering
- Uses Student-t distribution for fat tails

---

### ⚡ Jump-Diffusion Process

Used to capture sudden market shocks:

- Poisson-based jump arrival process
- Random jump sizes
- Models crashes and extreme market movements

---

## 📉 Backtesting Framework

The model is validated using industry-standard risk tests:

### 📊 Kupiec Test (Unconditional Coverage)
Evaluates whether the frequency of VaR violations matches the expected level.

### 🔁 Christoffersen Test (Conditional Coverage)
Evaluates:

- Independence of violations
- Clustering of risk events

---

## 📊 Outputs

The system generates:

- Forecast price distributions
- VaR and CVaR across multiple horizons
- Rolling forecast error analysis
- VaR violation timelines
- Monte Carlo simulation paths
- Risk heatmaps and summary tables

---

## 📈 Sample Results (NVDA)

- Observations: 1590
- VaR violations: 8.67%
- Kupiec Test: ❌ Reject H0 (risk underestimation detected)
- Christoffersen Test: mixed results
- Final Verdict: ⭐ ACCEPTABLE

---

## ▶️ How to Run

```bash
pip install numpy pandas matplotlib scipy arch
python jump_gjr_garch.py