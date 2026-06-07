import logging
import time

import pandas as pd

from config.output_config import (
    BACKTESTING_DATA,
    OUTPUT_PATH,
    PORTFOLIO_RETURNS,
    TRANSACTION_COST,
)
from utils.data_handler import load_data_parquet, save_data_csv

logger = logging.getLogger(__name__)


def compute_monthly_returns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compute the end-of-month returns for the predicted and actual values

    Args:
        predictions (pd.DataFrame): df of predicted and actual returns

    Returns:
        pd.DataFrame: end-of-month returns for the predicted and actual values
    """

    df = predictions.reset_index()

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")

    df = df.sort_values("date").groupby(["month", "ticker"]).last().reset_index()

    return df[["month", "ticker", "y_test", "y_pred"]]


def assign_quantiles(df: pd.DataFrame) -> pd.DataFrame:
    """Function that assigns each predicted return to a quantile (1 = bottom, 5 = top) to execute the profolio strategy

    Args:
        df (pd.DataFrame): monthly returns df

    Returns:
        pd.DataFrame: monthly retruns with quantiles
    """

    df["quantile"] = df.groupby("month")["y_pred"].transform(
        lambda x: pd.qcut(x, q=5, labels=[1, 2, 3, 4, 5])
    )

    return df


def compute_portfolio_returns(df: pd.DataFrame, transaction_cost: float) -> pd.Series:
    """Long-Short portfolio construction following the quantile rankings. The top 20% go long and the bottom  20% short. The retruns of the strategy include a fixed transactional cost of 15BPS for each round trip.

    Args:
        df (pd.DataFrame): quantile ranking df
        transaction_cost (float): transaction cost

    Returns:
        pd.Series: portfolio net returns
    """

    long_returns = df[df["quantile"] == 5].groupby("month")["y_test"].mean()
    short_returns = df[df["quantile"] == 1].groupby("month")["y_test"].mean()

    gross_returns = long_returns - short_returns
    net_returns = gross_returns - (2 * transaction_cost)

    logger.info(
        f"- Long leg => mean: {long_returns.mean():.4f} | std: {long_returns.std():.4f}"
    )
    logger.info(
        f"- Short leg => mean: {short_returns.mean():.4f} | std: {short_returns.std():.4f}"
    )
    logger.info(
        f"- Gross L/S => mean: {gross_returns.mean():.4f} | std: {gross_returns.std():.4f}"
    )

    return net_returns


def run_portfolio_pipeline(
    backtest_filename: str = BACKTESTING_DATA,
    output_dir: str = OUTPUT_PATH,
    transaction_cost: float = TRANSACTION_COST,
    portfolio_returns_file_name: str = PORTFOLIO_RETURNS,
) -> str:
    logger.info("Portfolio construction started")
    t0 = time.time()

    # 0. Load realised and predicted returns
    returns = load_data_parquet(backtest_filename, output_dir)

    # 1. Compute monthly returns
    monthly_returns = compute_monthly_returns(returns)

    logger.info("Monthly returns: ")
    logger.info(f"- Total observations:{len(monthly_returns)}")
    logger.info(f"- Unique months:{monthly_returns['month'].nunique()}")
    logger.info(f"- Unique tickers:{monthly_returns['ticker'].nunique()}")
    logger.info(
        f"- Date range:{monthly_returns['month'].min()} -> {monthly_returns['month'].max()}"
    )

    tickers_per_month = monthly_returns.groupby("month")["ticker"].count()
    logger.info("Tickers per month:")
    logger.info(f"- mean: {tickers_per_month.mean():.0f}")
    logger.info(f"- min: {tickers_per_month.min()}")
    logger.info(f"- max: {tickers_per_month.max()}")

    # 2. Rank predicted returns
    monthly_returns = assign_quantiles(monthly_returns)

    quantile_pred = monthly_returns.groupby("quantile")["y_pred"].mean()
    quantile_test = monthly_returns.groupby("quantile")["y_test"].mean()

    logger.info("Returns Ranking: ")
    logger.info("=> Quantile mean y_pred (should increase Q1 -> Q5):")
    for q, v in quantile_pred.items():
        logger.info(f"  - Q{q}: {v:.6f}")

    logger.info("=> Quantile mean y_test (realised returns by quantile):")
    for q, v in quantile_test.items():
        logger.info(f"  - Q{q}: {v:.6f}")

    # 3. Build Porfolio
    logger.info("Portfolio Metrics: ")

    net_returns = compute_portfolio_returns(monthly_returns, transaction_cost)

    logger.info(f"- Best month: {net_returns.max():.4f} ({net_returns.idxmax()})")
    logger.info(f"- Worst month:  {net_returns.min():.4f} ({net_returns.idxmin()})")

    logger.info("=> Top 5 months:")
    for month, val in net_returns.nlargest(5).items():
        logger.info(f"  - {month}: {val:.4f}")

    logger.info("=> Bottom 5 months:")
    for month, val in net_returns.nsmallest(5).items():
        logger.info(f"  - {month}: {val:.4f}")

    best_month = net_returns.idxmax()
    best_month_df = monthly_returns[monthly_returns["month"] == best_month]

    logger.info(f"Best month details - ({best_month}):")
    logger.info(
        f"- Tickers in long leg:  {len(best_month_df[best_month_df['quantile'] == 5])}"
    )
    logger.info(
        f"- Tickers in short leg: {len(best_month_df[best_month_df['quantile'] == 1])}"
    )

    logger.info(
        f"- Long leg return: {best_month_df[best_month_df['quantile'] == 5]['y_test'].mean():.4f}"
    )
    logger.info(
        f"- Short leg return: {best_month_df[best_month_df['quantile'] == 1]['y_test'].mean():.4f}"
    )
    logger.info(
        f"- Max single y_test in long: {best_month_df[best_month_df['quantile'] == 5]['y_test'].max():.4f}"
    )
    logger.info(
        f"- Min single y_test in short: {best_month_df[best_month_df['quantile'] == 1]['y_test'].min():.4f}"
    )

    logger.info("Additional Info:")
    logger.info(f"- NaNs in y_pred: {monthly_returns['y_pred'].isna().sum()}")
    logger.info(f"- NaNs in y_test: {monthly_returns['y_test'].isna().sum()}")
    logger.info(f"- NaNs in quantile: {monthly_returns['quantile'].isna().sum()}")

    # 4. Save data
    net_returns_df = net_returns.to_frame("net_portfolio_returns")
    save_data_csv(net_returns_df, portfolio_returns_file_name, output_dir)

    elapsed_t = time.time() - t0

    msg = f"Portfolio Construction completed - (time elapsed: {elapsed_t:.2f})"

    return msg
