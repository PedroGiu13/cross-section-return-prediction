# Cross-Sectional Return Prediction Using Machine Learning

## Overview

This project implements an end-to-end, modular machine learning pipeline for cross-section equity return prediction. The entire framework tries to emulate the entire workflow of an industry environment with the limitations of working with free, open-source resources. The goal is to determine whether a small set of price-based factors (momentum, change in momentum, max single day returns, and realised volatility), constructed with strict no-leakage temporal discipline, can generate statistically meaningful cross-sectional signals on a freely available equity dataset.

This project implements an end-to-end, modular machine learning pipeline for cross-sectional equity return prediction. The framework emulates a quantitative/ML workflow, from raw price ingestion to backtested portfolio performance, sing only free, open-source resources. The central research question is whether a minimal set of price-based factors (momentum, change in momentum, and realised volatility), constructed with strict no-look-ahead temporal discipline, can generate statistically meaningful cross-sectional signals on a freely available equity dataset.

---

## Key Results

| Metric | XGBoost |
|---|---|
| OOS R² | **0.9136%**
| Monthly IC mean | **0.0574**
| Monthly IC std | **0.1350**
| ICIR | **0.4249**
| Net Sharpe ratio | 0.301
| Annualised Volatility | 26.02%
| Max drawdown | -30.39%
| Calmar ratio | 0.259
| DSR (N=30, T=59) | 0.076
| Positive months | 59.3%

The model demonstrates genuine predictive content: a positive out-of-sample R² and a mean IC of 0.057 are consistent with findings in the empirical asset pricing literature for parsimonious factor sets. The portfolio Sharpe ratio reflects the structural challenge of running a market-neutral strategy over the 2021–2025 test window, which spans a bull market (2021), a sharp rate-driven bear market (2022), and a prolonged recovery (2023–2025). The Deflated Sharpe Ratio accounts for multiple-testing bias across 30 Optuna trials, providing a conservative but honest assessment of the signal's statistical significance given the 59-month test window.

---

## Pipeline Architecture

The project is structured as a modular, sequential pipeline where each stage owns its own data read/write, communicates through files on disk, and can be re-run independently. This design mirrors the separation-of-concerns principle used in production systems, where data, signal, and execution layers are decoupled.

Each module writes its outputs as Parquet or CSV files, so intermediate results persist across partial re-runs. The pipeline can be executed end-to-end with a single `python -m src.main` call or stage-by-stage for debugging and iteration.

```
main.py
  │
  ├── Data ingestion
  ├── Feature engineering
  ├── Model training
  ├── Model evaluation
  ├── Portfolio construction
  └── Backtesting
```

---

## Methodology

### Data
- **Universe:** 200 U.S. large-cap equities (`yfinance`)
- **Sample period:** January 2000 – December 2025
- **Train / Val / Test split:**
  - Train: January 2001 – December 2018
  - Validation: January 2019 – December 2020
  - Test (out-of-sample): January 2021 – November 2025

### Features

Three price-based factors constructed with strict temporal discipline — all rolling windows use only data available at prediction time $t$:

| Feature | Description |
|---|---|
| Momentum | Short-term (mom_1) and medium-term momentum (mom_6, mom_12). Estimated as cumulative log return over 252 days lagged 21 days |
| Change in Momentum | Difference between recent and lagged 6-month momentum |
| Maximum Daily Return | Maximum single-day return over a period of 21 days |
| Realised Volatility | Annualised standard deviation of daily returns over a 20-day and 60-day rolling window |

### Target Variable

The target is the next-month cumulative log return, computed as:

$$y_{i,t} = \sum_{k=2}^{22} \log\frac{P_{i,t+k}}{P_{i,t+k-1}}$$

This 21-day forward log return is shifted forward by −22 days (rather than −21) to introduce a clean one-trading-day gap between the feature formation period and the start of the return measurement window. This gap prevents any information leakage from the most recent close price appearing in both the feature and the target.

### Model

The model chosen for the prediction is the XGBoost, given its well-documented performance on tabular financial data and their built-in regularisation mechanisms that help mitigate overfitting on noisy cross-sectional targets.

- **Objective:** `reg:squarederror` — direct regression on next-month log return
- **Hyperparameter tuning:** Optuna with 30 trials of Bayesian (TPE) optimisation, evaluated on validation-set RMSE
- **Early stopping:** 50-round patience on validation RMSE, preventing overfitting during boosting
- **Temporal split:** strict walk-forward — no data from the validation or test windows ever enters the training set, eliminating lookahead bias at the modelling stage

### Portfolio Construction

At each month-end, the model generates a predicted return score for all 200 stocks in the universe. Stocks are ranked by predicted score and sorted into quintiles. A long-short portfolio is formed by going long the top quintile and short the bottom quintile, capturing the spread between the model's highest- and lowest-conviction predictions.

- **Long leg:** top quintile (Q5) — equal-weighted — 40 stocks
- **Short leg:** bottom quintile (Q1) — equal-weighted — 40 stocks
- **Rebalancing frequency:** monthly (aligned with prediction frequency)
- **Transaction costs:** 15 bps per leg (30 bps round-trip) applied at each rebalance

### Evaluation Metrics

- **OOS R²** — relative to zero forecast benchmark
- **Monthly IC** — cross-sectional Spearman rank correlation, averaged across months
- **ICIR** — IC mean / IC standard deviation
- **Sharpe ratio** — annualised, net of transaction costs
- **Maximum drawdown** — peak-to-trough on cumulative net return series
- **Calmar ratio** — annualised return / |max drawdown|
- **Deflated Sharpe Ratio** — corrected for multiple testing, non-normality, and sample length, with N=30 trials and T=59 test months

---

## Findings and Limitations

**What works well:** The model generates a positive out-of-sample R² and a mean IC of 0.057, consistent with the signal quality reported in the asset pricing ML literature for similar models. The quintile spread is monotonic, meaning Q5 consistently outperforms Q1 over the test period, confirming that the model's cross-sectional ranking contains economically meaningful information. The signal holds across the 2022 bear market and the 2023–2025 recovery, suggesting limited regime sensitivity for the ranking component.

**Key limitations:** The net Sharpe ratio is not particularly optima, a structural headwind for market-neutral strategies between 2021 and 2025. In addition, the Deflated Sharpe Ratio reflects that the observed Sharpe is insufficient to clear the multiple-testing threshold given 30 hyperparameter trials and only 59 months of out-of-sample data.

**Improvments:** To improve the DSR a longer OOS test window would be required. Moreover, the most direct improvement path is expanding the factor set to include volume-based signals and illiquidity measures, as this would likely improve IC consistency and widen the Q5–Q1 spread.

---

## Project Structure

```
├── src/
│   ├── main.py
│   ├── data_ingestion.py
│   ├── features.py
│   ├── model_training.py
│   ├── model_eval.py
│   ├── portfolio.py
│   └── backtest.py
│
├── config/
│   ├── data_config.py
│   ├── model_config.py
│   ├── output_config.py
│   └── backtest_config.py
│
├── utils/
│   ├── data_handler.py
│   └── model_handler.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
```

---

## Technical Stack

| Component | Tool |
|---|---|
| Data source | `yfinance` |
| Data processing | `pandas`, `numpy` |
| Model | `xgboost` |
| Hyperparameter tuning | `optuna` |
| Storage format | Parquet (`pyarrow`), CSV |
| Statistical tests | `scipy.stats` |
| Code quality | `pre-commit` hooks (`ruff`, `black`) |
---
