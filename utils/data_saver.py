import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_data(data: pd.DataFrame, file_name: str, cache_dir: str) -> None:

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
