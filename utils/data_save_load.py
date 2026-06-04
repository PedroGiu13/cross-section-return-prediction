"""Data saving and loading utilities for the cross-sectional factors model."""

import logging
from pathlib import Path

import pandas as pd

from utils.config import PROCESSED_DATA_PATH, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def save_raw_data(
    df: pd.DataFrame, filename: str, cache_dir: str = RAW_DATA_PATH
) -> None:
    """Save raw data to the RAW_DATA_PATH directory.

    Args:
        df (pd.DataFrame): DataFrame containing the raw data to save.
        filename (str): Name of the file (without extension) to save the data as.

    Raises:
        IOError: If there is an issue saving the file.
    """
    # Define parent directory and ensure it exists
    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{filename}.parquet"
    file_path = file_dir / file_name

    if file_path.exists():
        logger.warning(f"File {file_path} already exists and will be overwritten.")

    else:
        logger.info(f"Saving raw data to {file_path}")

        try:
            df.to_parquet(file_path, engine="pyarrow", index=True, compression="snappy")
            logger.info(f"Raw data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving raw data: {e}")
            raise OSError(f"Failed to save raw data: {e}")


def save_processed_data(
    df: pd.DataFrame, filename: str, cache_dir: str = PROCESSED_DATA_PATH
) -> None:
    """Save processed data to the PROCESSED_DATA_PATH directory.

    Args:
        df (pd.DataFrame): DataFrame containing the processed data to save.
        filename (str): Name of the file (without extension) to save the data as.

    Raises:
        IOError: If there is an issue saving the file.
    """
    # Define parent directory and ensure it exists
    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{filename}.parquet"
    file_path = file_dir / file_name

    if file_path.exists():
        logger.warning(f"File {file_path} already exists and will be overwritten.")

    else:
        logger.info(f"Saving processed data to {file_path}")

        try:
            df.to_parquet(file_path, engine="pyarrow", index=True, compression="snappy")
            logger.info(f"Processed data saved successfully to {file_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise OSError(f"Failed to save processed data: {e}")


def load_data(filename: str, cache_dir: str) -> pd.DataFrame:
    """Load data from a specified directory.

    Args:
        filename (str): Name of the file (without extension) to load.
        cache_dir (str): Directory to load the file from (RAW_DATA_PATH or PROCESSED_DATA_PATH).
    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """

    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    file_name = f"{filename}.parquet"
    file_path = file_dir / file_name

    if not file_path.exists():
        logger.error(f"File {file_path} does not exist.")
        raise FileNotFoundError(f"File {file_path} not found.")

    df = pd.read_parquet(file_path)
    logger.info(f"Data loaded successfully from {file_path}. Shape: {df.shape}")

    return df
