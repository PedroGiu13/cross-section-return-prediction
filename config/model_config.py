MODEL_DIR = "models/"

FEATURE_LIST = [
    "ticker",
    "mom_1",
    "mom_6",
    "momm_12",
    "maxret",
    "chmom",
    "retvol_20",
    "retvol_60",
    "log_returns",
]

XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "eta": 0.05,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "lambda": 1.0,
    "alpha": 0.1,
    "seed": 42,
}
