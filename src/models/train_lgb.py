import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.config import LGB_PARAMS, MODEL_DIR, VALID_START, TEST_START, RIDGE_ALPHA
from src.features.build_features import get_base_features, get_cs_features
from src.evaluation.metrics import calc_spread_return_sharpe, rank_prediction, spearman_corr


def split_time_series(df, valid_start=VALID_START, test_start=TEST_START):
    train = df[df["Date"] < valid_start].copy()
    valid = df[(df["Date"] >= valid_start) & (df["Date"] < test_start)].copy()
    test = df[df["Date"] >= test_start].copy()
    print(f"Train: {train['Date'].min().date()} ~ {train['Date'].max().date()}, {len(train):,} rows")
    print(f"Valid:  {valid['Date'].min().date()} ~ {valid['Date'].max().date()}, {len(valid):,} rows")
    print(f"Test:   {test['Date'].min().date()} ~ {test['Date'].max().date()}, {len(test):,} rows")
    return train, valid, test


def evaluate_prediction(df, pred_col, split_name=""):
    df = df.dropna(subset=["Target"]).copy()
    if len(df) == 0:
        return 0.0, 0.0
    df = rank_prediction(df, pred_col=pred_col)
    sharpe, _ = calc_spread_return_sharpe(df, rank_col="Rank", target_col="Target")
    sp = spearman_corr(df, pred_col=pred_col, target_col="Target")
    print(f"  {split_name}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}, "
          f"days={df['Date'].nunique()}, rows={len(df):,}")
    return sharpe, sp


def train_stacking(df, base_feats=None, cs_feats=None, lgb_rounds=1000):
    if base_feats is None:
        base_feats = get_base_features()
    if cs_feats is None:
        cs_feats = get_cs_features()

    base_feats = [f for f in base_feats if f in df.columns]
    cs_feats = [f for f in cs_feats if f in df.columns]
    print(f"Base features ({len(base_feats)}): {base_feats}")
    print(f"CS rank features ({len(cs_feats)}): {cs_feats}")

    train_df, valid_df, test_df = split_time_series(df)
    train_df = train_df.dropna(subset=["Target"])
    valid_df = valid_df.dropna(subset=["Target"])
    test_df = test_df.dropna(subset=["Target"])

    # === Level 0: Ridge on base features ===
    print("\n--- Level 0: Ridge Regression ---")
    X_train_base = train_df[base_feats].fillna(0)
    y_train = train_df["Target"]

    ridge = Ridge(alpha=RIDGE_ALPHA)
    ridge.fit(X_train_base, y_train)

    for split_name, split_df in [("Train", train_df), ("Valid", valid_df), ("Test", test_df)]:
        split_df = split_df.copy()
        split_df["pred_ridge"] = ridge.predict(split_df[base_feats].fillna(0))
        evaluate_prediction(split_df, "pred_ridge", split_name)

    # Ridge predictions for residual computation
    train_df = train_df.copy()
    valid_df = valid_df.copy()
    test_df = test_df.copy()
    train_df["ridge_pred"] = ridge.predict(X_train_base)
    valid_df["ridge_pred"] = ridge.predict(valid_df[base_feats].fillna(0))
    test_df["ridge_pred"] = ridge.predict(test_df[base_feats].fillna(0))
    train_df["residual"] = train_df["Target"] - train_df["ridge_pred"]

    # === Level 1: LightGBM on residuals ===
    print("\n--- Level 1: LightGBM on Ridge residuals ---")
    X_train_cs = train_df[cs_feats].fillna(0)
    y_residual = train_df["residual"]

    dtrain = lgb.Dataset(X_train_cs, label=y_residual)
    dvalid = lgb.Dataset(
        valid_df[cs_feats].fillna(0),
        label=valid_df["Target"] - valid_df["ridge_pred"],
        reference=dtrain,
    )

    params = LGB_PARAMS.copy()
    print(f"Training LightGBM, {lgb_rounds} rounds, {len(cs_feats)} features...")

    gbm = lgb.train(
        params, dtrain,
        num_boost_round=lgb_rounds,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(200)],
    )
    print(f"Best iteration: {gbm.best_iteration}")

    # === Evaluate Level 1 only (GBT residual predictions) ===
    valid_df["gbm_residual"] = gbm.predict(valid_df[cs_feats].fillna(0))
    test_df["gbm_residual"] = gbm.predict(test_df[cs_feats].fillna(0))
    print("\n  GBT residuals only (no Ridge):")
    evaluate_prediction(valid_df, "gbm_residual", "Valid")
    evaluate_prediction(test_df, "gbm_residual", "Test")

    # === Stacked prediction ===
    valid_df["pred_stack"] = valid_df["ridge_pred"] + valid_df["gbm_residual"]
    test_df["pred_stack"] = test_df["ridge_pred"] + test_df["gbm_residual"]

    print("\n=== Stacked Model (Ridge + GBT) ===")
    for split_name, split_df, pred_col in [
        ("Valid", valid_df, "pred_stack"),
        ("Test", test_df, "pred_stack"),
    ]:
        evaluate_prediction(split_df, pred_col, split_name)

    # Per-year breakdown
    print("\nPer-year breakdown:")
    for year in [2021, 2022]:
        sub = test_df[test_df["Date"].dt.year == year].copy()
        if len(sub) > 0:
            evaluate_prediction(sub, "pred_stack", f"Test {year}")

    # Feature importance
    print("\nFeature Importance (GBT):")
    imp = pd.Series(gbm.feature_importance(), index=cs_feats).sort_values(ascending=False)
    for f, v in imp.items():
        print(f"  {f}: {v}")

    # Save models
    ridge_path = MODEL_DIR / "ridge_stacking.joblib"
    joblib.dump((ridge, base_feats), str(ridge_path))
    print(f"\nRidge saved: {ridge_path}")

    gbm_path = MODEL_DIR / "gbm_stacking.txt"
    gbm.save_model(str(gbm_path))
    print(f"GBT saved: {gbm_path}")

    # Also save cs_feats for inference
    with open(str(MODEL_DIR / "cs_features.txt"), "w") as f:
        f.write("\n".join(cs_feats))

    return {
        "ridge": ridge,
        "gbm": gbm,
        "base_feats": base_feats,
        "cs_feats": cs_feats,
        "valid_df": valid_df,
        "test_df": test_df,
    }


def predict_stacking(df, ridge, gbm, base_feats, cs_feats):
    df = df.copy()
    df["ridge_pred"] = ridge.predict(df[base_feats].fillna(0))
    df["gbm_residual"] = gbm.predict(df[cs_feats].fillna(0))
    df["pred"] = df["ridge_pred"] + df["gbm_residual"]
    return df
