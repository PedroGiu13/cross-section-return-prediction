"""
Entry point for the cross-sectional factor model pipeline.

Usage:
    python main.py

Each stage of the pipeline corresponds to a GitHub issue:
    Issue #1 - Data ingestion
    Issue #2 - Feature engineering
    Issue #3 - Model training
    Issue #4 - Portfolio construction
    Issue #5 - Backtesting
"""

import logging

from config.backtest_config import PERIODS_YEAR, RISK_FREE_RATE
from config.data_config import CONFIG, PROCESSED_DATA_PATH, PROCESSED_TICKER_PRICE
from config.model_config import (
    EARLY_STOP,
    N_TRIALS,
    NUM_BOOST,
    TEST_START_DATE,
    VAL_START_DATE,
)
from config.output_config import OUTPUT_PATH, PORTFOLIO_RETURNS
from src.backtest import run_bactest_pipeline
from src.data_ingestion import run_ingestion_pipeline
from src.features import run_feature_pipeline
from src.model_eval import run_model_eval_pipeline
from src.model_training import run_model_building_pipeline
from src.portfolio import run_portfolio_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # 1. Fetch, Validate, and Process data
    ingestion_result = run_ingestion_pipeline(
        tickers=CONFIG["tickers"],
        start_date=CONFIG["start"],
        end_date=CONFIG["end"],
        max_nan_pct=CONFIG["max_nan_pct"],
        min_history=CONFIG["min_history"],
    )

    print(ingestion_result)

    # 2. Compute Feature Matrix
    feature_matrix_result = run_feature_pipeline(
        PROCESSED_TICKER_PRICE, PROCESSED_DATA_PATH
    )

    print(feature_matrix_result)

    # 3.  Compute the model
    x_test, y_test = run_model_building_pipeline(
        VAL_START_DATE, TEST_START_DATE, NUM_BOOST, EARLY_STOP, N_TRIALS
    )

    # 4. Model Evaluation
    model_eval_results = run_model_eval_pipeline(x_test, y_test)
    print(model_eval_results)

    # 5. Portfolio Construction
    net_returns = run_portfolio_pipeline()
    print(net_returns)

    # 6. Backtest
    backtest_result = run_bactest_pipeline(
        PERIODS_YEAR,
        RISK_FREE_RATE,
        N_TRIALS,
        PORTFOLIO_RETURNS,
        OUTPUT_PATH,
    )
