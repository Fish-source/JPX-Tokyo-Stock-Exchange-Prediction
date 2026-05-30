import numpy as np
import pandas as pd
from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_feature_columns
from src.models.train_lgb import evaluate_model
import lightgbm as lgb

df = load_merged(use_train=True, use_supplement=True)
df = preprocess(df)
df = build_all_features(df, use_cache=True)
feat = get_feature_columns(df)

test_df = df[df["Date"] >= "2021-07-01"].dropna(subset=["Target"]).copy()
train_df = df[df["Date"] < "2021-07-01"].dropna(subset=["Target"]).copy()

configs = [
    {
        "label": "v1: lr=0.01, leaves=31, depth=6, l1=1, l2=1, 2000r",
        "params": {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.01, "num_leaves": 31, "max_depth": 6,
            "feature_fraction": 0.6, "bagging_fraction": 0.6, "bagging_freq": 1,
            "lambda_l1": 1.0, "lambda_l2": 1.0, "min_child_samples": 500,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        },
        "n_rounds": 2000,
    },
    {
        "label": "v2: lr=0.02, leaves=63, depth=-1, l1=0.5, l2=0.5, 1500r",
        "params": {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
            "feature_fraction": 0.5, "bagging_fraction": 0.5, "bagging_freq": 1,
            "lambda_l1": 0.5, "lambda_l2": 0.5, "min_child_samples": 300,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        },
        "n_rounds": 1500,
    },
    {
        "label": "v3: lr=0.005, leaves=15, depth=5, l1=2, l2=2, 3000r",
        "params": {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.005, "num_leaves": 15, "max_depth": 5,
            "feature_fraction": 0.7, "bagging_fraction": 0.7, "bagging_freq": 1,
            "lambda_l1": 2.0, "lambda_l2": 2.0, "min_child_samples": 800,
            "verbose": -1, "n_jobs": -1, "seed": 42,
        },
        "n_rounds": 3000,
    },
    {
        "label": "v4: lr=0.03, leaves=31, depth=7, l1=0, l2=0.5, 1000r, extra_trees",
        "params": {
            "objective": "regression", "metric": "rmse",
            "learning_rate": 0.03, "num_leaves": 31, "max_depth": 7,
            "feature_fraction": 0.5, "bagging_fraction": 0.5, "bagging_freq": 1,
            "lambda_l1": 0.0, "lambda_l2": 0.5, "min_child_samples": 200,
            "verbose": -1, "n_jobs": -1, "seed": 42, "extra_trees": True,
        },
        "n_rounds": 1000,
    },
]

best_sharpe = -999
best_label = ""
for cfg in configs:
    print(f"\n{'='*60}")
    print(f"  {cfg['label']}")
    print(f"{'='*60}")
    dtrain = lgb.Dataset(train_df[feat], label=train_df["Target"])
    model = lgb.train(
        cfg["params"], dtrain,
        num_boost_round=cfg["n_rounds"],
        callbacks=[lgb.log_evaluation(500)],
    )

    test_df_copy = test_df.copy()
    test_df_copy["pred"] = model.predict(test_df_copy[feat])
    from src.evaluation.metrics import rank_prediction, calc_spread_return_sharpe, spearman_corr
    test_df_copy = rank_prediction(test_df_copy, pred_col="pred")
    sharpe, daily = calc_spread_return_sharpe(test_df_copy, rank_col="Rank", target_col="Target")
    sp = spearman_corr(test_df_copy, pred_col="pred", target_col="Target")
    print(f"  Test: Sharpe={sharpe:.4f}, Spearman={sp:.4f}, days={len(daily)}")

    if sharpe > best_sharpe:
        best_sharpe = sharpe
        best_label = cfg["label"]
        model.save_model(str("output/models/lgb_best.txt"))
        print(f"  >>> New best model saved!")

print(f"\n\nBest: {best_label}, Sharpe={best_sharpe:.4f}")
