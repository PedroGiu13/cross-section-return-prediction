# Files and directories
MODEL_DIR = "models/"
FEAT_MATRIX_FILE_NAME = "feature_matrix"
XGB_BEST_PARMS_FILE_NAME = "xgb_best_params"
XGB_MODEL_FILE_NAME = "xgb_model"

# Temporal split
VAL_START_DATE = ""
TEST_START_DATE = ""

# Features
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


# XGBoost Training params
NUM_BOOST = 1000
EARLY_STOP = 30
N_TRIALS = 60
