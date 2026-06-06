import logging
import time

import pandas as pd
import yfinance as yf

from config.data_config import (
    PROCESSED_DATA_PATH,
    PROCESSED_TICKER_PRICE,
    RAW_DATA_PATH,
    RAW_TICKER_PRICE,
)
from utils.data_handler import save_data_parquet

logger = logging.getLogger(__name__)


def get_ticker_data(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame:
    """Function data retreives the daily adjusted closing price of a given ticker universe from yahoo finance.

    Args:
        tickers (list[str]): universe of tickers to fetch.
        start_date (str): start data in 'YYYY-MM-DD' format.
        end_date (str): end data in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: df with tickers as columns and date as index
    """

    try:
        logger.info(f"Fetching data from {start_date} to {end_date}")
        data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            auto_adjust=True,
            progress=False,
        )

        if data is None or data.empty:
            logger.warning("Empty response from yfinance")
            return pd.DataFrame()

        df = data["Close"]

        if isinstance(df, pd.Series):
            return df.to_frame(tickers[0])

        all_nan_tickers = df.columns[df.isna().all()].tolist()
        if all_nan_tickers:
            logger.warning(
                f"Dropping {len(all_nan_tickers)} tickers with no data: {all_nan_tickers}"
            )
            df = df.drop(columns=all_nan_tickers)

        return df

    except Exception as e:
        logger.error(f"Failed to fetch data from yfinance: {e}")
        return pd.DataFrame()


def validate_data(data: pd.DataFrame, max_nan_pct: float, min_history: int) -> bool:
    """Function that validates data integrity.

    Tests:
        - If df is empty
        - If null values exceed a threshold
        - If there are duplicate time indexes
        - If the trading days history of the ticker satisfies the minimum threshold

    Args:
        data (pd.DataFrame): df with daily adjusted closing prices
        max_nan_pct (float, optional): Maximum pct of null values accepted.
        min_history (int, optional): Minimum number of trading days.

    Returns:
        bool:
            - True: if data is valid
            - False: if there's any validation issue
    """

    is_valid = True

    if data.empty:
        logger.error("DataFrame is empty")
        return False

    # Null Values
    per_ticker_nan_count = data.isna().sum()
    total_nan_count = per_ticker_nan_count.sum()

    if total_nan_count > 0:
        total_nan_pct = total_nan_count / data.size

        if total_nan_pct > max_nan_pct:
            logger.error(f"Null values threshold exceeded: {total_nan_pct:.1%}")
            is_valid = False
        else:
            logger.warning(
                f"Found {total_nan_count} null values. Below threshold: {total_nan_pct:.1%}"
            )

    # Data index
    if data.index.duplicated().any():
        n = data.index.duplicated().sum()
        logger.warning(f"{n} duplicate index where found")
        is_valid = False

    # Ticker history
    for ticker, days in data.notna().sum().items():
        if days < min_history:
            logger.warning(
                f"{ticker} does not achieve the minimum history of trading days: {days} < {min_history}"
            )

    return is_valid


def process_data(
    data: pd.DataFrame, max_nan_pct: float, min_history: int
) -> pd.DataFrame:
    """Function that process the data before computing the features.

    Args:
        data (pd.DataFrame): df with daily adjusted closing prices
        max_nan_pct (float, optional): Maximum pct of null values accepted. Defaults to 0.1.
        min_history (int, optional): Minimum number of trading days. Defaults to 252 (1 year).

    Returns:
        pd.DataFrame: clean df with daily adj. closing prices
    """

    nan_pct = data.isna().mean()
    trading_history = data.notna().sum()

    mask = (nan_pct < max_nan_pct) & (trading_history >= min_history)

    clean_data = data.loc[:, mask]

    dropped_data = data.shape[1] - clean_data.shape[1]
    if dropped_data:
        logger.info(f"Dropped {dropped_data} tickers.")

    return clean_data


def run_ingestion_pipeline(
    tickers: list[str],
    start_date: str,
    end_date: str,
    max_nan_pct: float,
    min_history: int,
    raw_data_path: str = RAW_DATA_PATH,
    raw_file_name: str = RAW_TICKER_PRICE,
    processed_data_path: str = PROCESSED_DATA_PATH,
    processed_file_name: str = PROCESSED_TICKER_PRICE,
) -> str:
    """Orchestrator function of the entire data ingestion process.

    Steps:
        1. Fetch data
        2. Save raw data
        3. Validate and process data
        4. Save processed data

    Args:
        tickers (list[str]): universe of tickers to fetch.
        start_date (str): start data in 'YYYY-MM-DD' format.
        end_date (str): end data in 'YYYY-MM-DD' format.
        raw_data_path (str, optional): file path to save raw data. Defaults to RAW_DATA_PATH.
        raw_file_name (str, optional): file name for raw data. Defaults to RAW_TICKER_PRICE.
        processed_data_path (str, optional): file path to save processed data. Defaults to PROCESSED_DATA_PATH.
        processed_file_name(str, optional): file name for processed data. Defaults to PROCESSED_TICKER_PRICE
        max_nan_pct (float, optional): Maximum pct of null values accepted. Defaults to 0.1.
        min_history (int, optional): Minimum number of trading days. Defaults to 252 (1 year)

    Returns:
        str: outcome of the ingestion pipeline
    """

    logger.info(
        f"Ingestion pipeline started fetching {len(tickers)} Tickers from {start_date} to {end_date}"
    )
    t0 = time.time()

    # 1. Fetch data
    df = get_ticker_data(tickers, start_date, end_date)

    # 2. Validate and clean data
    if df.empty:
        logger.error("No data fetched - aborting pipeline")
        return "Pipeline failed: no data fetched"

    # 3. Save raw data
    save_data_parquet(df, raw_file_name, raw_data_path)

    if not validate_data(df, max_nan_pct, min_history):
        logger.warning("Data Validation failed - Processing Data")

    df_clean = process_data(df, max_nan_pct, min_history)

    # 4. Save Processed Data
    save_data_parquet(df_clean, processed_file_name, processed_data_path)

    elapsed_t = time.time() - t0
    msg = f"Ingestion Pipline completed (time elapsed: {elapsed_t:.2f}"
    logger.info(msg)
    logger.info("Processed Data:")
    logger.info(f"- Rows: {df_clean.shape[0]}")
    logger.info(f"- Columns: {df_clean.shape[1]}")
    logger.info(f"- Observations: {df_clean.size}")
    logger.info(f"- NaN Count: {df_clean.isna().sum().sum()}")

    return msg
