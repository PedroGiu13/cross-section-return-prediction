"""
Entry point for the cross-sectional factor model pipeline.

Usage:
    python main.py

Each stage of the pipeline corresponds to a GitHub issue:
    Issue #1 - Data ingestion          (src/data_loader.py)
    Issue #2 - Feature engineering     (src/features.py)
    Issue #3 - Model training          (src/model.py)
    Issue #4 - Backtesting             (src/backtest.py)
    Issue #5 - Portfolio construction  (src/portfolio.py)
"""

import logging

from src.data_ingestion import run_ingestion_pipeline
from utils.config import CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    ingestion_result = run_ingestion_pipeline(
        tickers=CONFIG["tickers"],
        start_date=CONFIG["start"],
        end_date=CONFIG["end"],
        max_nan_pct=CONFIG["max_nan_pct"],
        min_history=CONFIG["min_history"],
    )
