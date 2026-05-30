import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "jpx-tokyo-stock-exchange-prediction"
TRAIN_DIR = DATA_DIR / "train_files"
SUPPLEMENT_DIR = DATA_DIR / "supplemental_files"
TEST_DIR = DATA_DIR / "example_test_files"

OUTPUT_DIR = ROOT_DIR / "output"
MODEL_DIR = OUTPUT_DIR / "models"
FEATURE_DIR = OUTPUT_DIR / "features"
SUBMISSION_DIR = OUTPUT_DIR / "submissions"

for d in [MODEL_DIR, FEATURE_DIR, SUBMISSION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TRAIN_START = "2017-01-04"
VALID_START = "2021-01-01"
TEST_START = "2021-07-01"

MOMENTUM_WINDOWS = [1, 5, 10, 20, 40, 60]
VOL_WINDOWS = [5, 10, 20, 60]

LGB_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "max_depth": 6,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "min_child_samples": 500,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

LGB_FIT_PARAMS = {
    "num_boost_round": 10000,
    "early_stopping_rounds": 500,
    "verbose_eval": 500,
}

FEATURE_CACHE_NAME = "features_v3.parquet"
