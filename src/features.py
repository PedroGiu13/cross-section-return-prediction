import logging
import time

import numpy as np
import pandas as pd

from config.data_config import PROCESSED_DATA_PATH, PROCESSED_TICKER_PRICE
from config.model_config import FEAT_MATRIX_FILE_NAME
from utils.data_handler import load_data_parquet, save_data_csv

logger = logging.getLogger(__name__)


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute the forward shifted log returns for the target variable using a 21 day rolling window and leaving a 1 day buffer.

    Args:
        prices (pd.DataFrame): daily adj closing prices df

    Returns:
        pd.DataFrame: df of log returns
    """

    ratio = prices / prices.shift(1)

    log_returns = ratio.apply(np.log)

    rolling_log_returns = log_returns.rolling(21).sum().shift(-22)

    return rolling_log_returns


def compute_mom(
    prices: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """compute momentum for:
        - 21 dyas (1 month) leaving 1 day buffer
        - 126 days (6 months) leaving 2 day buffer to avoid the reversal contamination of the most recent month
        - 252 days (12 months) leaving 2 day buffer to avoid the reversal contamination of the most recent month


    Args:
        prices (pd.DataFrame): daily adj closing prices df

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: momentum for 1, 6, 8, and 12 months
    """
    # Compute mom ratios
    ratio_mom_1 = prices.shift(1) / prices.shift(22)
    ratio_mom_6 = prices.shift(2) / prices.shift(128)
    ratio_mom_12 = prices.shift(2) / prices.shift(254)

    mom_1 = ratio_mom_1.apply(np.log)
    mom_6 = ratio_mom_6.apply(np.log)
    mom_12 = ratio_mom_12.apply(np.log)

    # Additional momentum for change in momentum
    ratio_mom_6_lag_8 = prices.shift(8) / prices.shift(134)
    mom_6_lag_8 = ratio_mom_6_lag_8.apply(np.log)

    return mom_1, mom_6, mom_6_lag_8, mom_12


def compute_maxret(prices: pd.DataFrame) -> pd.DataFrame:
    """Computes the Maximum Single-Day treturn over a rolling window of 21 trading days

    Args:
        prices (pd.DataFrame): daily adj closing prices df

    Returns:
        pd.DataFrame: maxret
    """

    ratio = prices / prices.shift(1)

    log_returns = ratio.apply(np.log)

    maxret = log_returns.rolling(21).max().shift(1)

    return maxret


def compute_chmom(mom_6: pd.DataFrame, mom_6_lag_8: pd.DataFrame) -> pd.DataFrame:
    """Computes change of momentum in the 6-month momentum from lag 2 and from lag 8. Indicates if the momentum is accelerating or deceleratring

    Args:
        mom_6 (pd.DataFrame): 6 month momentum

    Returns:
        pd.DataFrame: chmom
    """

    return mom_6 - mom_6_lag_8


def compute_retvol(prices: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Computes the annualised realised volatility over a 20-day rolling period and a 60-day rolling period. Each lagged one day. This factors capture short and medium-horizon risk.

    Args:
        prices (pd.DataFrame): daily adj closing prices df

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: realised volatility for 20 and 60 days
    """

    ratio = prices / prices.shift(1)
    log_returns = ratio.apply(np.log)

    retvol_20 = log_returns.rolling(20).std().shift(1) * np.sqrt(252)
    retvol_60 = log_returns.rolling(60).std().shift(1) * np.sqrt(252)

    return retvol_20, retvol_60


def transform_feature_matrix(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    long_feature_matrix = (
        feature_matrix.stack(level=1)
        .rename_axis(index=["date", "ticker"])
        .reset_index(level="ticker")
        .sort_index()
    )

    return long_feature_matrix


def run_feature_pipeline(
    load_file_name: str = PROCESSED_TICKER_PRICE,
    cache_dir: str = PROCESSED_DATA_PATH,
    save_file_name=FEAT_MATRIX_FILE_NAME,
) -> str:
    logger.info("Computing Feature Matrix")
    t0 = time.time()

    # 0. Load Data
    prices = load_data_parquet(load_file_name, cache_dir)

    if prices.empty:
        logger.error(f"Unable to load data for {load_file_name}")
        return f"Unable to load data for {load_file_name}"

    # 1. Compute Features:
    log_returns = compute_log_returns(prices)
    mom_1, mom_6, mom_6_lag_8, mom_12 = compute_mom(prices)
    maxret = compute_maxret(prices)
    chmom = compute_chmom(mom_6, mom_6_lag_8)
    retvol_20, retvol_60 = compute_retvol(prices)

    # 2. Create feature matrix:
    wide_feature_matrix = pd.concat(
        [mom_1, mom_6, mom_12, maxret, chmom, retvol_20, retvol_60, log_returns],
        axis=1,
        keys=[
            "mom_1",
            "mom_6",
            "mom_12",
            "maxret",
            "chmom",
            "retvol_20",
            "retvol_60",
            "log_returns",
        ],
    )

    wide_feature_matrix = wide_feature_matrix.dropna(how="any")

    feature_matrix = transform_feature_matrix(wide_feature_matrix)

    # 3. Save Feature Matrix
    save_data_csv(feature_matrix, save_file_name, cache_dir)

    elapsed_t = time.time() - t0

    msg = f"Feature matrix computed (time elapsed: {elapsed_t:.2f})"
    logger.info(msg)
    logger.info("Feature Matrix:")
    logger.info(f"- Rows: {feature_matrix.shape[0]}")
    logger.info(f"- Columns: {feature_matrix.shape[1]}")
    logger.info(f"- Observations: {feature_matrix.size}")
    logger.info(f"- NaN Count: {feature_matrix.isna().sum().sum()}")

    return msg
