import numpy as np
import pandas as pd
from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_feature_columns
import lightgbm as lgb
from src.evaluation.metrics import rank_prediction, calc_spread_return_sharpe, spearman_corr

df = load_merged(use_train=True, use_supplement=True)
df = preprocess(df)
df = build_all_features(df, use_cache=True)
feat = get_feature_columns(df)

test_df = df[df["Date"] >= "2021-07-01"].dropna(subset=["Target"]).copy()
train_df = df[df["Date"] < "2021-07-01"].dropna(subset=["Target"]).copy()

base_params = {
    "objective": "regression", "metric": "rmse",
    "learning_rate": 0.02, "num_leaves": 63, "max_depth": -1,
    "feature_fraction": 0.5, "bagging_fraction": 0.5, "bagging_freq": 1,
    "lambda_l1": 0.5, "lambda_l2": 0.5, "min_child_samples": 300,
    "verbose": -1, "n_jobs": -1, "seed": 42,
}

# Vary rounds
results = []
for n_rounds in [1000, 1500, 2000, 2500, 3000]:
    dtrain = lgb.Dataset(train_df[feat], label=train_df["Target"])
    model = lgb.train(base_params, dtrain, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(1000)])
    t = test_df.copy()
    t["pred"] = model.predict(t[feat])
    t = rank_prediction(t, pred_col="pred")
    sharpe, _ = calc_spread_return_sharpe(t, rank_col="Rank", target_col="Target")
    sp = spearman_corr(t, pred_col="pred", target_col="Target")
    print(f"rounds={n_rounds}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}")
    results.append((n_rounds, sharpe, sp))

best = max(results, key=lambda x: x[1])
print(f"\nBest: rounds={best[0]}, Sharpe={best[1]:.4f}")

# Save best model
best_rounds = best[0]
dtrain = lgb.Dataset(train_df[feat], label=train_df["Target"])
best_model = lgb.train(base_params, dtrain, num_boost_round=best_rounds, callbacks=[lgb.log_evaluation(500)])
best_model.save_model("output/models/lgb_best.txt")
print(f"Saved best model with {best_rounds} rounds")
