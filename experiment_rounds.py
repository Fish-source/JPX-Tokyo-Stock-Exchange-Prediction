import numpy as np
import pandas as pd
from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_feature_columns
from src.models.train_lgb import evaluate_model
from src.config import LGB_PARAMS
import lightgbm as lgb

df = load_merged(use_train=True, use_supplement=True)
df = preprocess(df)
df = build_all_features(df, use_cache=True)
feat = get_feature_columns(df)

test_df = df[df["Date"] >= "2021-07-01"].dropna(subset=["Target"]).copy()

for label, train_end, n_rounds in [
    ("ALL pre-test (1200r)", "2021-07-01", 1200),
    ("ALL pre-test (2000r)", "2021-07-01", 2000),
]:
    print(f"\n=== {label} ===")
    train_df = df[df["Date"] < train_end].dropna(subset=["Target"]).copy()
    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    dtrain = lgb.Dataset(train_df[feat], label=train_df["Target"])
    params = LGB_PARAMS.copy()
    model = lgb.train(params, dtrain, num_boost_round=n_rounds, callbacks=[lgb.log_evaluation(500)])

    evaluate_model(model, test_df, feat, "Test")

    imp = pd.DataFrame({
        "feature": feat,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("Top 10 features:")
    print(imp.head(10).to_string(index=False))
