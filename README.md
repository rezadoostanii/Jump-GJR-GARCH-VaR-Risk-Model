# 🚀 Jump + GJR-GARCH Risk Modeling & VaR Backtesting Engine

## 📌 Overview
This project implements a hybrid quantitative risk modeling system combining:

- GJR-GARCH(1,1) volatility model with leverage effect
- Student-t distributed returns (fat tails)
- Poisson jump-diffusion process (market shocks)
- Value-at-Risk (VaR) and Conditional VaR (CVaR)
- Full statistical backtesting framework (Kupiec & Christoffersen tests)
- Rolling window forecasting system

The goal is to build a realistic institutional-grade risk engine for financial assets.

---

## ⚙️ Key Features

- 📊 Volatility modeling with asymmetric shocks (GJR-GARCH)
- ⚡ Jump diffusion process for crash modeling
- 📉 VaR & CVaR estimation across multiple horizons
- 🔍 Backtesting with statistical validation
- 📈 Rolling forecast evaluation
- 🧪 Exception clustering detection
- 💾 SQLite-based financial data integration
- 📊 Visualization of risk, forecasts, and distributions

---

## 🧠 Models Used

### GJR-GARCH(1,1)
Captures volatility clustering and leverage effect:
- Negative shocks increase volatility more than positive ones
- Student-t distribution handles fat tails

---

### Jump Diffusion Model
Models sudden market movements:
- Poisson jump arrivals
- Random jump size distribution
- Captures extreme events (crashes / spikes)

---

## 📉 Backtesting Framework

### Kupiec Test (Unconditional Coverage)
Checks if VaR violation frequency matches expected level.

### Christoffersen Test (Conditional Coverage)
Tests:
- Independence of violations
- Clustering of risk events

### Outputs:
- Exception rate
- Likelihood ratio statistics
- P-values
- Model rejection/acceptance decision

---

## 📊 Outputs Generated

- Forecast price distributions
- VaR / CVaR risk curves
- Rolling prediction error analysis
- VaR violation timeline
- Monte Carlo simulation paths
- Risk summary tables
- Heatmaps across time horizons

---

## 📁 Project Structure

- main.py → Main model & execution script
- requirements.txt → Dependencies
- Result/ → Output charts (plots, heatmaps)
- README.md → Documentation

---

## ⚠️ Data Requirement

This repository does NOT include financial data.

You must provide your own SQLite database (portfolio.db).

### Required format:
- Table name: {ticker}_prices
- Columns:
  - Date
  - Close

Example table:
nvda_prices  
- Date  
- Close  

The system will automatically read the data from the database path defined in main.py.

---

## ▶️ How to Run

Install dependencies:
```bash
pip install -r requirements.txt
