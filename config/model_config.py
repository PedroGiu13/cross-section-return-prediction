# Files and directories
MODEL_DIR = "models/"
FEAT_MATRIX_FILE_NAME = "feature_matrix"
XGB_BEST_PARMS_FILE_NAME = "xgb_best_params"
XGB_MODEL_FILE_NAME = "xgb_model"
XGB_LEARNING_CURVE_FILE_NAME = "xgb_learning_curve"
# Temporal split
VAL_START_DATE = "2019-01-01"
TEST_START_DATE = "2021-01-01"

# Features
FEATURE_LIST = [
    "ticker",
    "mom_1",
    "mom_6",
    "mom_12",
    "maxret",
    "chmom",
    "retvol_20",
    "retvol_60",
    "log_returns",
]


# XGBoost Training params
FINE_TUNE = False  # Change to True for fine tunning
NUM_BOOST = 1000
EARLY_STOP = 30
N_TRIALS = 60
