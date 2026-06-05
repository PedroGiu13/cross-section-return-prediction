import json
import logging
import time
from pathlib import Path

import optuna
import pandas as pd
import xgboost as xgb
from optuna.samplers import TPESampler

from config.data_config import PROCESSED_DATA_PATH
from config.model_config import (
    FEAT_MATRIX_FILE_NAME,
    FEATURE_LIST,
    FINE_TUNE,
    MODEL_DIR,
    XGB_BEST_PARMS_FILE_NAME,
    XGB_LEARNING_CURVE_FILE_NAME,
    XGB_MODEL_FILE_NAME,
)
from utils.data_handler import load_data_csv, save_data_csv
from utils.model_handler import save_xgb_model

optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = logging.getLogger(__name__)


def data_split(
    dataset: pd.DataFrame, feat_list: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Split feature matrix into features (X) and target variable (y)

    Args:
        dataset (pd.DataFrame): feature matrix
        feat_list (list[str], optional): list of features

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


def train_model(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    params: dict,
    num_boost: int,
    early_stop: int,
) -> tuple[xgb.Booster, pd.DataFrame]:
    """Function that  train the XGBoost model with early stopping on validation RMSE

    Args:
        x_train (pd.DataFrame): training features
        x_val (pd.DataFrame): validation features
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        params (dict, optional): CGBoost hyperparameters.
        num_boost (int, optional): maximum number of boosting rounds.
        early_stop (int, optional): stopping criterion.

    Returns:
        tuple[xgb.Booster, dict]: fitted booster and evaluatioon results
    """

    # Compute DMatrices for memory efficiency
    d_train = xgb.DMatrix(x_train, label=y_train)
    d_val = xgb.DMatrix(x_val, label=y_val)

    # Compute model
    eval_results = {}

    model = xgb.train(
        params=params,
        dtrain=d_train,
        evals=[(d_train, "train"), (d_val, "val")],
        num_boost_round=num_boost,
        early_stopping_rounds=early_stop,
        evals_result=eval_results,
        verbose_eval=False,
    )

    logger.info(
        f"Training Complete - best iterations: {model.best_iteration}; best val RMSE: {model.best_score:.4f}"
    )

    eval_df = pd.DataFrame(
        {
            "round": range(len(eval_results["train"]["rmse"])),
            "train_rmse": eval_results["train"]["rmse"],
            "val_rmse": eval_results["val"]["rmse"],
        }
    )

    return model, eval_df


def hyperparameter_objective(
    trial: optuna.Trial,
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    num_boost: int,
    early_stop: int,
) -> float:
    """Function that defines the hyperparameter objectives before the tuning process

    Args:
        trial (optuna.Trial): process of evaluation
        x_train (pd.DataFrame): training features
        x_val (pd.DataFrame): validation features
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        num_boost (int, optional): maximum number of boosting rounds.
        early_stop (int, optional): stopping criterion.

    Returns:
        float: best RMSE
    """
    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 40,
        "eta": trial.suggest_float("eta", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.9),
        "max_depth": trial.suggest_int("max_depth", 3, 5),
        "lambda": trial.suggest_float("lambda", 1.0, 3.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.1, 1.0, log=True),
    }

    _, eval_results = train_model(
        x_train, x_val, y_train, y_val, params, num_boost, early_stop
    )

    best_val_rmse = min(eval_results["val"]["rmse"])
    return best_val_rmse


def hyperparameter_tuning(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    num_boost: int,
    early_stop: int,
    n_trials: int,
) -> dict:
    """Function to perform the parameter tuning

    Args:
        x_train (pd.DataFrame): training features
        x_val (pd.DataFrame): validation features
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        num_boost (int, optional): maximum number of boosting rounds.
        early_stop (int, optional): stopping criterion.
        n_trials (int): number of trials

    Returns:
        dict: optimal parameters
    """
    sampler = TPESampler(seed=40)

    study = optuna.create_study(
        direction="minimize", sampler=sampler, study_name="xgboost_pred_model"
    )

    study.optimize(
        lambda trial: hyperparameter_objective(
            trial, x_train, x_val, y_train, y_val, num_boost, early_stop
        ),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": 40,
        **study.best_params,
    }

    return best_params


def load_or_tune_params(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    num_boost: int,
    early_stop: int,
    n_trials: int,
    params_file_name: str,
    model_dir: str,
    retune: bool = FINE_TUNE,
) -> dict:
    """Function that checks if the model has been tuned or not. If the  optimal parameters are saved they are loaded. If not the tuner is executed

    Args:
        x_train (pd.DataFrame): training features
        x_val (pd.DataFrame): validation features
        y_train (pd.Series): training target
        y_val (pd.Series): validation target
        num_boost (int, optional): maximum number of boosting rounds.
        early_stop (int, optional): stopping criterion.
        n_trials (int): number of trials
        params_file_name (str): best params file name
        model_dir (str): model results directory
        retune (bool, optional): check to force retunning. Defaults to FINE_TUNE.

    Returns:
        dict: optimal parameters
    """
    file_dir = Path(model_dir)
    file_dir.mkdir(parents=True, exist_ok=True)

    model_name = f"{params_file_name}.json"

    file_path = file_dir / model_name

    if file_path.exists() and not retune:
        try:
            logger.info(f"Loading saved hyperparameters from {file_path}")
            with open(file_path) as f:
                best_params = json.load(f)
            return best_params

        except Exception as e:
            logger.warning(
                f"Failed to load params from {file_path}: {e}. Re-running tuning."
            )

    best_params = hyperparameter_tuning(
        x_train, x_val, y_train, y_val, num_boost, early_stop, n_trials
    )

    with open(file_path, "w") as f:
        json.dump(best_params, f, indent=2)
    logger.info(f"Hyperparameters saved to {file_path}")

    return best_params


def run_model_building_pipeline(
    val_start_date: str,
    test_start_date: str,
    num_boost: int,
    early_stop: int,
    n_trials: int,
    feat_list: list[str] = FEATURE_LIST,
    feat_file_name: str = FEAT_MATRIX_FILE_NAME,
    cache_dir: str = PROCESSED_DATA_PATH,
    params_file_name: str = XGB_BEST_PARMS_FILE_NAME,
    model_file_name: str = XGB_MODEL_FILE_NAME,
    learning_curve_file_name: str = XGB_LEARNING_CURVE_FILE_NAME,
    model_dir: str = MODEL_DIR,
) -> tuple[pd.DataFrame, pd.Series]:
    """_summary_

    Args:
        val_start_date (str): _description_
        test_start_date (str): _description_
        num_boost (int): _description_
        early_stop (int): _description_
        n_trials (int): _description_
        feat_list (list[str], optional): _description_. Defaults to FEATURE_LIST.
        feat_file_name (str, optional): _description_. Defaults to FEAT_MATRIX_FILE_NAME.
        cache_dir (str, optional): _description_. Defaults to PROCESSED_DATA_PATH.
        params_file_name (str, optional): _description_. Defaults to XGB_BEST_PARMS_FILE_NAME.
        model_file_name (str, optional): _description_. Defaults to XGB_MODEL_FILE_NAME.
        learning_curve_file_name (str, optional): _description_. Defaults to XGB_LEARNING_CURVE_FILE_NAME.
        model_dir (str, optional): _description_. Defaults to MODEL_DIR.

    Returns:
        tuple[pd.DataFrame, pd.Series]: _description_
    """
    logger.info("Starting XGBoost training")
    t0 = time.time()

    # 0.  Load feature  matrix
    feature_matrix = load_data_csv(feat_file_name, cache_dir)

    # 1. Split Data
    x, y = data_split(feature_matrix, feat_list)

    # 2. Split train/val/test
    x_train, x_val, x_test, y_train, y_val, y_test = temporal_split(
        x, y, val_start_date, test_start_date
    )

    logger.info(f"Train split - from {x_train.index.min()} to {x_train.index.max()}")
    logger.info(f"Val split - from {x_val.index.min()} to {x_val.index.max()}")
    logger.info(f"Test split - from {x_test.index.min()} to {x_test.index.max()}")

    # 3. Train model
    # If params are not tuned  -> tune params -> train model
    # If params are tuned -> train model
    params = load_or_tune_params(
        x_train,
        x_val,
        y_train,
        y_val,
        num_boost,
        early_stop,
        n_trials,
        params_file_name,
        model_dir,
    )

    model, eval_results = train_model(
        x_train, x_val, y_train, y_val, params, num_boost, early_stop
    )

    # 4. Save model and evaluation results
    save_xgb_model(model, model_file_name, model_dir)
    save_data_csv(eval_results, learning_curve_file_name, model_dir)

    elapsed_t = time.time() - t0
    msg = f"Model Trained (time elapsed: {elapsed_t:.2f})"
    print(msg)

    return x_test, y_test
