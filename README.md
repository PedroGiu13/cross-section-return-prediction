# Cross-Sectional Return Prediction Using Machine Learning

## Overview

This project implements an end-to-end, modular machine learning pipeline for cross-sectiopn equity return prediction. The entire framework tries to emulate the entire workflow of an industry environment with the limitations of working with free, open-source resources. The goal is to determine whether a small set of price-based factors (momentum, change in momentum, max single day returns, and realised volatility), constructed with strict no-leakage temporal discipline, can generate statistically meaningful cross-sectional signals on a freely available equity dataset.

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
| DSR (N=60, T=59) | 0.076
| Positive months | 59.3%



The model demonstrates genuine predictive content with good OOS R² and IC metrics for the XGBoost model, while portfolio Sharpe reflects the additional difficulty of a longer 59-month test window spanning multiple distinct market regimes (2021 bull, 2022 bear, 2023–2025 recovery).

---

## Pipeline Architecture

The project is structured as a modular end-to-end pipeline where each stage owns its own data read/write and communicates through files on disk.

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

- **Universe:** 200 U.S. large-cap equities sourced from Yahoo Finance via `yfinance`
- **Sample period:** January 2000 – December 2025
- **Train / Val / Test split:**
  - Train: January 2001 – December 2018
  - Validation: January 2019 – December 2020
  - Test (out-of-sample): January 2021 – November 2025

### Features

Three price-based factors constructed with strict temporal discipline — all rolling windows use only data available at prediction time $t$:

| Feature | Description |
|---|---|
| 12-1 Momentum | Cumulative log return over 252 days lagged 21 days |
| Change in Momentum | Difference between recent and lagged 6-month momentum |
| Realised Volatility | Annualised standard deviation of daily returns over a 60-day rolling window |

### Target Variable

Next-month cumulative log return:

$$y_{i,t} = \sum_{k=2}^{22} \log\frac{P_{i,t+k}}{P_{i,t+k-1}}$$

Computed as a 21-day rolling log return shifted −22 days forward, ensuring a clean one-trading-day gap between the feature formation period and the start of the prediction window.

### Model

- **Architecture:** XGBoost gradient-boosted trees (`reg:squarederror`)
- **Hyperparameter tuning:** Optuna (60 trials, Bayesian optimisation) on the validation set
- **Early stopping:** patience 50 rounds on validation RMSE
- **Temporal split:** strict walk-forward — no data from validation or test windows enters training

### Portfolio Construction

At each month-end, all 200 stocks are ranked by predicted return. A long-short quintile portfolio is formed:

- **Long:** top quintile (Q5) — equal weighted — 40 stocks
- **Short:** bottom quintile (Q1) — equal weighted — 40 stocks
- **Rebalancing:** monthly
- **Transaction costs:** 15 bps per leg (30 bps round-trip)

### Evaluation Metrics

- **OOS R²** — relative to zero forecast benchmark (Gu et al., 2020)
- **Monthly IC** — cross-sectional Spearman rank correlation, averaged across months
- **ICIR** — IC mean / IC standard deviation
- **Sharpe ratio** — annualised, net of transaction costs
- **Maximum drawdown** — peak-to-trough on cumulative net return series
- **Calmar ratio** — annualised return / |max drawdown|
- **Deflated Sharpe Ratio** — corrected for multiple testing, non-normality, and sample length (Bailey & López de Prado, 2014), with N=60 trials and T=59 test months

---

## Project Structure

```
├── main.py                        # pipeline entry point
│
├── src/
│   ├── data_ingestion.py          # fetch, validate, normalise, save
│   ├── features.py                # factor construction, assembly, stacking
│   ├── model_training.py          # temporal split, scaling, XGBoost training, tuning
│   ├── model_eval.py              # OOS R², IC, feature importance
│   ├── portfolio.py               # quintile construction, long-short returns
│   └── backtest.py                # Sharpe, drawdown, Calmar, DSR
│
├── config/
│   ├── data_config.py             # tickers, dates, file paths
│   └── model_config.py            # hyperparameters, split dates, tuning config
│
├── utils/
│   └── data_handler.py            # save_data(), load_data() helpers
│
├── data/
│   ├── raw/                       # OHLCV prices from yfinance
│   └── processed/                 # feature matrix (long format)
│
├── models/                        # trained XGBoost model + fitted scaler
│
├── results/                       # metrics, predictions, portfolio returns
│
└── logs/                          # structured pipeline run logs
```

---

## Technical Stack

| Component | Tool |
|---|---|
| Data source | `yfinance` |
| Data processing | `pandas`, `numpy` |
| Model | `xgboost` |
| Hyperparameter tuning | `optuna` |
| Serialisation | `joblib` (scaler), XGBoost native JSON (model) |
| Storage format | Parquet (`pyarrow`), CSV |
| Logging | Python `logging` module — structured, per-module |
| Statistical tests | `scipy.stats` |

---

## Findings and Limitations

**What works well.** The model generates positive out-of-sample R² and IC metrics consistent with the asset pricing ML literature. The quintile spread is monotonic — stocks ranked in the top quintile consistently outperform those in the bottom quintile over the test period. The signal is positive across both bear (2022) and recovery (2023–2025) regimes.

**Key limitations.** The net Sharpe ratio (0.363) falls below the paper's benchmark (0.799), driven primarily by the short leg consistently earning positive returns in a largely bullish test environment — a structural challenge for market-neutral strategies over 2021–2025. The Deflated Sharpe Ratio of 0.059 reflects that a Sharpe of 0.363 is insufficient to clear the multiple-testing threshold after 60 hyperparameter trials, consistent with the paper's own finding that 60+ months of out-of-sample data are required for reliable DSR classification. Expanding the factor set beyond three features — adding volume, illiquidity, and additional momentum windows as in the original paper — would likely improve signal quality and portfolio Sharpe.

---

## References

- Lama, D. (2024). *Cross-Sectional Return Prediction Using Machine Learning: A Practitioner Pipeline.*
- Gu, S., Kelly, B., & Xiu, D. (2020). Empirical asset pricing via machine learning. *Review of Financial Studies*, 33(5), 2223–2273.
- Bailey, D. H., & López de Prado, M. (2014). The deflated Sharpe ratio. *Journal of Portfolio Management*, 40(5), 94–107.
- Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers. *Journal of Finance*, 48(1), 65–91.
- López de Prado, M. (2018). *Advances in Financial Machine Learning.* Wiley.

---

## Author

**[Your Name]**
[LinkedIn] · [GitHub] · [Email]
