import xgboost as xgb
import numpy as np
import pandas as pd
from src.config import XGB_PARAMS, MODEL_DIR, VALID_START, TEST_START
from src.features.build_features import get_feature_columns, get_categorical_features
from src.evaluation.metrics import calc_spread_return_sharpe, rank_prediction, spearman_corr
from src.models.train_lgb import calc_time_weights


def train_xgboost(df, feature_cols=None, params=None, n_rounds=3000, use_early_stopping=True):
    if params is None:
        params = XGB_PARAMS.copy()
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    num_cols = [c for c in feature_cols if c != "SecuritiesCode" and c != "CodeBin"]

    train_df = df[(df["Date"] >= "2019-01-01") & (df["Date"] < VALID_START)].dropna(subset=["Target"])
    valid_df = df[(df["Date"] >= VALID_START) & (df["Date"] < TEST_START)].dropna(subset=["Target"])
    test_df = df[df["Date"] >= TEST_START].dropna(subset=["Target"])

    print(f"XGBoost train: {len(train_df)} rows, valid: {len(valid_df)} rows, test: {len(test_df)} rows")

    X_train = train_df[num_cols].fillna(0)
    y_train = train_df["Target"]
    X_valid = valid_df[num_cols].fillna(0)
    y_valid = valid_df["Target"]

    w_train = calc_time_weights(train_df)

    dtrain = xgb.DMatrix(X_train, label=y_train, weight=w_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    print(f"Training XGBoost, {n_rounds} rounds...")

    if use_early_stopping:
        model = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            early_stopping_rounds=300,
            verbose_eval=200,
        )
    else:
        model = xgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            evals=[(dtrain, "train"), (dvalid, "valid")],
            verbose_eval=200,
        )

    model_path = MODEL_DIR / "xgb_model.json"
    model.save_model(str(model_path))
    print(f"XGBoost model saved: {model_path}")

    print("Evaluation:")
    _eval_xgb(model, valid_df, num_cols, "Valid")
    _eval_xgb(model, test_df, num_cols, "Test")

    importance = model.get_score(importance_type="gain")
    importance = pd.DataFrame([
        {"feature": k, "importance": v} for k, v in importance.items()
    ]).sort_values("importance", ascending=False)
    print("\nFeature importance Top 20:")
    print(importance.head(20).to_string(index=False))

    return model, importance, num_cols


def _eval_xgb(model, df, feature_cols, split_name=""):
    df = df.dropna(subset=["Target"]).copy()
    if len(df) == 0:
        return
    dtest = xgb.DMatrix(df[feature_cols].fillna(0))
    df["pred"] = model.predict(dtest)
    df = rank_prediction(df, pred_col="pred")
    sharpe, daily_returns = calc_spread_return_sharpe(df, rank_col="Rank", target_col="Target")
    sp = spearman_corr(df, pred_col="pred", target_col="Target")
    print(f"  {split_name}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}")


def load_xgb_model(model_name="xgb_model.json"):
    model_path = MODEL_DIR / model_name
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model
