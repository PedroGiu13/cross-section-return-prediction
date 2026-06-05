import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_data_parquet(data: pd.DataFrame, file_name: str, cache_dir: str) -> None:
    """Function to save data as parquet to the given folder

    Args:
        data (pd.DataFrame): df of data to save
        file_name (str): file name
        cache_dir (str): directory to save file
    """

    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{file_name}.parquet"
    file_path = file_dir / file_name

    if file_path.exists():
        logger.info(f"File already exists: {file_path}")

    else:
        try:
            data.to_parquet(file_path, engine="pyarrow", compression="snappy")
            logger.info(f"File succesfully saved - {file_path}")

        except Exception as e:
            logger.error(f"Unable to save file: {e}")


def save_data_csv(data: pd.DataFrame, file_name: str, cache_dir: str) -> None:
    """Function to save data as csv to the given folder

    Args:
        data (pd.DataFrame): df of data to save
        file_name (str): file name
        cache_dir (str): directory to save file
    """

    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{file_name}.csv"
    file_path = file_dir / file_name

    if file_path.exists():
        logger.info(f"File already exists: {file_path}")

    else:
        try:
            data.to_csv(file_path, index=True)
            logger.info(f"File succesfully saved - {file_path}")

        except Exception as e:
            logger.error(f"Unable to save file: {e}")


def load_data_parquet(file_name: str, cache_dir: str) -> pd.DataFrame:
    """Function to load parquet file from a specified folder

    Args:
        file_name (str): file name
        cache_dir (str): directory to save file
    """

    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{file_name}.parquet"
    file_path = file_dir / file_name

    if file_path.exists():
        try:
            df = pd.read_parquet(file_path)
            logger.info(f"{file_name} succesfully loaded")
            return df

        except Exception as e:
            logger.error(f"Unable to load file: {e}")
            return pd.DataFrame()
    else:
        logger.error(f"{file_name} does not exists")
        return pd.DataFrame()


def load_data_csv(file_name: str, cache_dir: str) -> pd.DataFrame:
    """Function to load csv file from a specified folder

    Args:
        file_name (str): file name
        cache_dir (str): directory to save file
    """

    file_dir = Path(cache_dir)
    file_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"{file_name}.csv"
    file_path = file_dir / file_name

    if file_path.exists():
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            logger.info(f"{file_name} succesfully loaded")
            return df

        except Exception as e:
            logger.error(f"Unable to load file: {e}")
            return pd.DataFrame()
    else:
        logger.error(f"{file_name} does not exists")
        return pd.DataFrame()
