import logging

import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from config.model_config import FEATURE_LIST, MODEL_DIR, XGBOOST_PARAMS

logger = logging.getLogger(__name__)


def data_split(
    dataset: pd.DataFrame, feat_list: list[str] = FEATURE_LIST
) -> tuple[pd.DataFrame, pd.Series]:
    """Split feature matrix into features (X) and target variable (y)

    Args:
        dataset (pd.DataFrame): feature matrix
        feat_list (list[str], optional): list of features. Defaults to FEATURE_LIST.

    Returns:
        tuple[pd.DataFrame, pd.Series]: X and y
    """

    for feat in feat_list:
        if feat not in dataset.columns:
            logger.warning(f"Missing feature: {feat}")

    x = dataset.drop(["ticker", "log_returns"], axis=1)
    y = dataset["log_returns"]

    return x, y


def temporal_split(
    x: pd.DataFrame, y: pd.Series, val_start_date: str, test_start_date: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Compute the temporal split of X and y into train, validation, and test sets.

    Args:
        x (pd.DataFrame): feature matrix
        y (pd.Series): target variab;e
        val_start_date (str): train split
        test_start_date (str): test split

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]: X_train,  X_val, X_test, y_train, y_val, y_test
    """

    # X split
    x_train = x[x.index < val_start_date]
    x_val = x[(x.index >= val_start_date) & (x.index < test_start_date)]
    x_test = x[x.index >= test_start_date]

    # y split
    y_train = y[y.index < val_start_date]
    y_val = y[(y.index >= val_start_date) & (y.index < test_start_date)]
    y_test = y[y.index >= test_start_date]

    return x_train, x_val, x_test, y_train, y_val, y_test


def data_scaler(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    model_dir: str = MODEL_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Start scaler
    scaler = StandardScaler()
    scaler.set_output(transform="pandas")

    # Scale Data
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_val_scaled, x_test_scaled  # type: ignore


def train_model(
    x_train_scaled: pd.DataFrame,
    y_train: pd.Series,
    params: dict = XGBOOST_PARAMS,
    num_boost_round: int = 1000,
    early_stop_rounds: int = 50,
) -> tuple[xgb.Booster, dict]:
    # Compute DMatrices for memory efficiency
    d_train = xgb.DMatrix(x_train_scaled, label=y_train)

    # Compute model
    eval_results = {}

    model = xgb.train(
        params=params,
        dtrain=d_train,
        num_boost_round=num_boost_round,
        early_stop_rounds=early_stop_rounds,
        eval_results=eval_results,
        verbose=False,
    )

    logger.info(
        f"Training Complete - best iterations: {model.best_iteration}; best val RMSE: {model.best_score:.4f}"
    )

    return model, eval_results


# def hyperparameter_tunning(
#     trial: optuna.Trial,
#     X_train_scaled: pd.DataFrame,
#     X_val_scaled: pd.DataFrame,
#     y_train: pd.Series,
#     y_val: pd.Series,
# ) -> float:

#     params = {
#         "objective": "reg:squarederror",
#         "eval_metric": "rmse",
#         "alpha": 0.1,
#         "lambda": 1,
#         "seed": 40,
#         "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
#         "subsample": trial.suggest_float("subsample", 0.7, 0.9),
#     }

#     model, eval_results = train_model(
#         X_train_scaled, X_val_scaled, y_train, y_val, params
#     )

#     val_rmse = min(eval_results["val"]["rmse"])
#     return val_rmse


# def run_model_building_pipeline():
#     # Save Scaler
#     file_name = f"{model_dir}/scaler_v1.pkl"
#     joblib.dump(scaler, file_name)
