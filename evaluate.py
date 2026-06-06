import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_base_features, get_cs_features
from src.evaluation.metrics import calc_spread_return_sharpe, rank_prediction, spearman_corr
from src.models.train_lgb import predict_stacking
from src.config import MODEL_DIR


def main():
    print("=" * 60)
    print("Stacking Model Evaluation")
    print("=" * 60)

    print("\nLoading data...")
    df = load_merged(use_train=True, use_supplement=True, use_secondary=True)
    df = preprocess(df)
    df = build_all_features(df, use_cache=True)

    ridge_path = MODEL_DIR / "ridge_stacking.joblib"
    gbm_path = MODEL_DIR / "gbm_stacking.txt"
    cs_feat_path = MODEL_DIR / "cs_features.txt"

    if not ridge_path.exists() or not gbm_path.exists():
        print("Models not found! Run run_pipeline.py first.")
        return

    ridge, base_feats = joblib.load(str(ridge_path))
    gbm = lgb.Booster(model_file=str(gbm_path))
    with open(str(cs_feat_path)) as f:
        cs_feats = [l.strip() for l in f if l.strip()]

    base_feats = [f for f in base_feats if f in df.columns]
    cs_feats = [f for f in cs_feats if f in df.columns]
    print(f"Base: {len(base_feats)} features, CS: {len(cs_feats)} features")

    # Evaluate per period
    periods = [
        ("2019-01-01", "2020-01-01", "2019"),
        ("2020-01-01", "2021-01-01", "2020"),
        ("2021-01-01", "2021-07-01", "2021-H1"),
        ("2021-07-01", "2022-01-01", "2021-H2"),
        ("2022-01-01", "2023-01-01", "2022"),
    ]

    print("\nPer-period results:")
    for start, end, label in periods:
        sub = df[(df["Date"] >= start) & (df["Date"] < end)].dropna(subset=["Target"]).copy()
        if len(sub) < 1000:
            continue
        sub = predict_stacking(sub, ridge, gbm, base_feats, cs_feats)
        sub = rank_prediction(sub, pred_col="pred")
        sharpe, _ = calc_spread_return_sharpe(sub, rank_col="Rank", target_col="Target")
        sp = spearman_corr(sub, pred_col="pred", target_col="Target")
        print(f"  {label}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}, "
              f"days={sub['Date'].nunique()}")

    # Full valid + test
    print("\nFull period results:")
    valid = df[(df["Date"] >= "2021-01-01") & (df["Date"] < "2021-07-01")].dropna(subset=["Target"]).copy()
    test = df[df["Date"] >= "2021-07-01"].dropna(subset=["Target"]).copy()

    for name, sub in [("Valid", valid), ("Test", test)]:
        sub = predict_stacking(sub, ridge, gbm, base_feats, cs_feats)
        sub = rank_prediction(sub, pred_col="pred")
        sharpe, _ = calc_spread_return_sharpe(sub, rank_col="Rank", target_col="Target")
        sp = spearman_corr(sub, pred_col="pred", target_col="Target")
        print(f"  {name}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}")

    # Per-month breakdown
    print("\nPer-month test breakdown:")
    test = predict_stacking(test, ridge, gbm, base_feats, cs_feats)
    test = rank_prediction(test, pred_col="pred")
    test["YearMonth"] = test["Date"].dt.to_period("M")
    for ym, group in test.groupby("YearMonth"):
        group = group.dropna(subset=["Target", "Rank"])
        if len(group) < 100:
            continue
        s, _ = calc_spread_return_sharpe(group, rank_col="Rank", target_col="Target")
        print(f"  {ym}: Sharpe={s:.4f}, days={group['Date'].nunique()}")


if __name__ == "__main__":
    main()
