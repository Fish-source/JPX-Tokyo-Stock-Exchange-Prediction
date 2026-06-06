import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("JPX_DATA_DIR", str(ROOT_DIR / "jpx-tokyo-stock-exchange-prediction")))
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

RETURN_WINDOWS = [1, 5, 10, 20]

RIDGE_ALPHA = 100

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.01,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "min_child_samples": 100,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}

FEATURE_CACHE_NAME = "features_v6.parquet"

BASE_FEATURES = [
    "Open_z", "High_z", "Low_z", "Close_z", "Volume_z",
    "Daily_Range_z", "Mean_z", "SupervisionFlag", "ExpectedDividend",
]

CS_RANK_FEATURES = [
    "Return_1d_rank", "Return_5d_rank", "Return_10d_rank", "Return_20d_rank",
    "Volatility_20d_rank", "TargetLag_1d_rank",
    "Close_z_rank", "Volume_z_rank", "IntradayRange_rank",
]
