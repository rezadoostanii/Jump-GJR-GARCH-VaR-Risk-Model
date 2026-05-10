# 🚀 Jump + GJR-GARCH VaR Risk Model

## 📌 Overview

This project is a **quantitative risk modeling engine** for financial assets.  
It combines volatility modeling, jump processes, and statistical risk measurement to estimate and backtest market risk.

The system is designed for **market risk analysis and portfolio risk research**.

---

## ⚙️ Models Used

### 📊 GJR-GARCH(1,1)
- Models volatility clustering in financial returns
- Captures **leverage effect** (bad news increases volatility more than good news)
- Uses Student-t distribution for fat tails

### ⚡ Jump Diffusion Model
- Simulates sudden market shocks (crashes/spikes)
- Poisson jump arrivals
- Random jump size distribution

---

## 📉 Risk Measures

- Value-at-Risk (VaR)
- Conditional Value-at-Risk (CVaR)
- Multi-horizon risk forecasts (30, 60, 90, 120 days)

---

## 🔍 Backtesting

The model is validated using:

### ✔ Kupiec Test
- Checks if VaR violation frequency is correct
- Detects under/overestimation of risk

### ✔ Christoffersen Test
- Checks independence of violations
- Detects clustering of extreme events

---

## 📊 Outputs

The system generates:

- Monte Carlo price simulations
- VaR & CVaR tables
- Risk heatmaps
- Forecast distributions
- Rolling backtest results
- Exception analysis plots

---

## 🧠 Key Insights

- GJR-GARCH captures volatility clustering effectively
- Jump component improves tail risk modeling
- VaR alone is insufficient without jump modeling
- CVaR provides more stable downside risk measure
- Model performance improves under rolling validation

---

## 🛠️ Tech Stack

- Python 3
- NumPy / Pandas
- SciPy
- ARCH package
- Matplotlib
- SQLite

---

## ▶️ How to Run

```bash

pip install numpy pandas matplotlib scipy arch
python main.py
⚠️ Important Note

This repository does NOT include financial data due to size and licensing constraints.  
Users must supply their own SQLite database with the following structure:

- Table: {ticker}_prices
- Columns: Date, Close

The system is designed to work with any asset once properly formatted.
