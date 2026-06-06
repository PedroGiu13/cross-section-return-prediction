import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from config.eval_config import (
    BACKTESTING_DATA,
    EVAL_METRICS_FILE_NAME,
    FEATURE_IMPORTANCE_FILE_NAME,
    OUTPUT_PATH,
)
from config.model_config import MODEL_DIR, XGB_MODEL_FILE_NAME
from utils.data_handler import save_data_csv, save_data_parquet

logger = logging.getLogger(__name__)


def make_prediction(model: xgb.Booster, x_test: pd.DataFrame) -> pd.Series:
    """Function that makes the prediction using the trained XGBoost  model

    Args:
        model (xgb.Booster): XGBoost model
        x_test (pd.DataFrame): test set

    Returns:
        pd.Series: target predictions
    """

    d_test = xgb.DMatrix(x_test)

    y_pred = model.predict(d_test)

    return y_pred  # type: ignore


def compute_oos_r2(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Computes the OOS R^2 relative to a zero forecast.

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values

    Returns:
        float: OOS R^2
    """

    model_mse = np.mean((y_true - y_pred) ** 2)

    zero_mse = np.mean(y_true**2)

    return float(1 - (model_mse / zero_mse))


def compute_ic(y_true: pd.Series, y_pred: pd.Series) -> dict:
    """Computes monthly Spearman's Information Coefficient between predictions and actual returns

    Args:
        y_true (pd.Series): true target values
        y_pred (pd.Series): predicted target values

    Returns:
        dict: IC metrics and series
    """

    target_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    target_df = target_df.reset_index()
    target_df["month"] = pd.to_datetime(target_df["date"]).dt.to_period("M")

    ic_series = (
        target_df.groupby("month").apply(
            lambda x: x["y_true"].corr(x["y_pred"], method="spearman")
        )
    ).dropna()

    ic_mean = ic_series.mean()
    ic_std = ic_series.std()

    return {
        "ic_mean": round(ic_mean, 6),
        "ic_std": round(ic_std, 6),
        "icir": round(ic_mean / ic_std, 6),
        "ic_series": ic_series,
    }


def compute_feature_importance(model: xgb.Booster) -> pd.DataFrame:
    """Compute the normalised feature importance of the  XGBoost model

    Args:
        model (xgb.Booster): Trained XGBoost model

    Returns:
        pd.DataFrame: Normalised Feature Importance df
    """

    scores = model.get_score(importance_type="gain")

    if not scores:
        logger.warning("No  scores returned")

    feature_importance = pd.Series(scores)
    feature_importance = feature_importance / feature_importance.sum()
    feature_importance = feature_importance.sort_values(ascending=False)
    feature_importance.index.name = "feature"
    feature_importance.name = "gain_normalised"

    return feature_importance.to_frame()


def run_model_eval_pipeline(
    x_test: pd.DataFrame,
    y_test: pd.Series,
    model_file_name: str = XGB_MODEL_FILE_NAME,
    model_dir: str = MODEL_DIR,
    feat_imp_file_name: str = FEATURE_IMPORTANCE_FILE_NAME,
    eval_metrics_file_name: str = EVAL_METRICS_FILE_NAME,
    backtest_file_name: str = BACKTESTING_DATA,
    output_dir: str = OUTPUT_PATH,
) -> dict:
    logger.info("Starting Model Evaluation")

    # 1. Import model
    model_path = f"{model_dir}{model_file_name}.json"

    model = xgb.Booster()
    model.load_model(model_path)

    # 2. Make Predictions
    y_pred = make_prediction(model, x_test)

    backtest_df = pd.DataFrame({"y_test": y_test, "y_pred": y_pred})

    # 3. Compute metrics
    oos_r2 = compute_oos_r2(y_test, y_pred)
    ic_results = compute_ic(y_test, y_pred)

    metrics = {
        "oos_r2": oos_r2,
        "ic_mean": ic_results["ic_mean"],
        "ic_std": ic_results["ic_std"],
        "icir": ic_results["icir"],
    }

    feature_importance_df = compute_feature_importance(model)
    # metrics_df = pd.DataFrame(metrics)

    # 4. Save data
    save_data_parquet(backtest_df, backtest_file_name, output_dir)
    save_data_csv(feature_importance_df, feat_imp_file_name, output_dir)
    # save_data_csv(metrics_df, eval_metrics_file_name, output_dir)

    return metrics
