import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import norm

from utils.data_handler import load_data_csv

logger = logging.getLogger(__name__)


def compute_return_metrics(
    returns: pd.Series, periods_per_year: int, risk_free_rate: float
) -> dict:
    """Computes metrics that can be directly estimated from the returns"
    - Annualised Log Returns
    - Annualised Volatility
    - Sharpe Ratio
    - Win-Loss Ratio

    Args:
        returns (pd.Series): long-short portfolio net returns
        periods_per_year (int): periods in a year
        risk_free_rate (float): risk free rate

    Returns:
        dict: dictionary of metrics
    """

    ann_return = returns.mean() * periods_per_year
    ann_std = returns.std() * np.sqrt(periods_per_year)
    sharpe_ratio = (ann_return - risk_free_rate) / ann_std

    avg_win = returns[returns > 0].mean()
    avg_loss = returns[returns < 0].mean()

    return {
        "ann_return": round(ann_return, 4),
        "ann_std": round(ann_std, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "positive_months": round((returns > 0).mean(), 4),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "win_loss_ratio": round(abs(avg_win / avg_loss), 4),
    }


def compute_drawdown(returns: pd.Series) -> dict:
    # Compute Max Drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Compute Calmar Ratio from Drawdown
    ann_ret = returns.mean() * 12
    calmar = ann_ret / abs(max_drawdown)

    return {
        "max_drawdown": round(max_drawdown, 4),
        "calmar_ratio": round(calmar, 4),
        "cumulative_returns": cum_returns,
        "drawdown_series": drawdown,
    }


def compute_dsr(
    returns: pd.Series, sharp_ratio: int, n_trials: int, periods_per_year: int
) -> float:
    t = len(returns)
    skew = returns.skew()
    kurt = returns.kurtosis()

    logger.info("DSR inputs:")
    logger.info(f"- Return skewness: {returns.skew():.4f}")
    logger.info(f"- Return kurtosis: {returns.kurtosis():.4f}")
    logger.info(f"- SR per period: {sharp_ratio / np.sqrt(12):.4f}")

    expected_max_sr = (
        (1 - np.euler_gamma) * norm.ppf(1 - 1 / n_trials)
        + np.euler_gamma * norm.ppf(1 - 1 / (n_trials * np.e))
    ) / np.sqrt(t - 1)

    sr_per_period = sharp_ratio / np.sqrt(periods_per_year)

    dsr = norm.cdf(
        (sr_per_period - expected_max_sr)
        * np.sqrt(t - 1)
        / np.sqrt(1 - skew * sr_per_period + ((kurt + 1) / 4) * sr_per_period**2)  # type: ignore
    )

    return float(round(dsr, 4))  # type: ignore


def run_bactest_pipeline(
    periods_per_year: int,
    risk_free_rate: float,
    n_trials: int,
    portfolio_returns_file_name: str,
    output_path: str,
) -> str:
    logger.info("Backtest pipeline started")
    t0 = time.time()

    # 0. Get net portfolio returns
    returns_df = load_data_csv(portfolio_returns_file_name, output_path)

    returns = returns_df["net_portfolio_returns"]

    logger.info(f"Evaluating {len(returns)} monthly observations")

    # 1. Compute Metrics
    return_metrics = compute_return_metrics(returns, periods_per_year, risk_free_rate)
    drawdown_metrics = compute_drawdown(returns)
    dsr = compute_dsr(
        returns, return_metrics["sharpe_ratio"], n_trials, periods_per_year
    )

    logger.info(f"Annualised return:{return_metrics['ann_return']:.2%}")
    logger.info(f"Annualised vol: {return_metrics['ann_std']:.2%}")
    logger.info(f"Sharpe ratio: {return_metrics['sharpe_ratio']:.3f}")
    logger.info(f"Max drawdown: {drawdown_metrics['max_drawdown']:.2%}")
    logger.info(f"Calmar ratio: {drawdown_metrics['calmar_ratio']:.3f}")
    logger.info(f"Positive months: {return_metrics['positive_months']:.1%}")
    logger.info(f"DSR (N={n_trials}, T={len(returns)}): {dsr:.3f}")

    # Save Metrics
    # metrics = {
    #     **return_metrics,
    #     "max_drawdown": drawdown_metrics["max_drawdown"],
    #     "calmar_ratio": drawdown_metrics["calmar_ratio"],
    #     "dsr": dsr,
    #     "n_trials": n_trials,
    #     "n_months": len(returns),
    # }

    elapsed_t = time.time() - t0

    msg = f"Backtesting complited - (time elapsed: {elapsed_t:.2f})"
    logger.info(msg)
    return msg
