import lightgbm as lgb
import numpy as np
import pandas as pd
from src.config import LGB_PARAMS, LGB_FIT_PARAMS, MODEL_DIR, VALID_START, TEST_START
from src.features.build_features import get_feature_columns
from src.evaluation.metrics import calc_spread_return_sharpe, rank_prediction, spearman_corr


def split_time_series(df, valid_start=VALID_START, test_start=TEST_START):
    train = df[df["Date"] < valid_start].copy()
    valid = df[(df["Date"] >= valid_start) & (df["Date"] < test_start)].copy()
    test = df[df["Date"] >= test_start].copy()
    print(f"训练集: {train['Date'].min().date()} ~ {train['Date'].max().date()}, {len(train)} 行")
    print(f"验证集: {valid['Date'].min().date()} ~ {valid['Date'].max().date()}, {len(valid)} 行")
    print(f"测试集: {test['Date'].min().date()} ~ {test['Date'].max().date()}, {len(test)} 行")
    return train, valid, test


def evaluate_model(model, df, feature_cols, split_name=""):
    df = df.dropna(subset=["Target"]).copy()
    if len(df) == 0:
        return
    df["pred"] = model.predict(df[feature_cols])
    df = rank_prediction(df, pred_col="pred")
    sharpe, daily_returns = calc_spread_return_sharpe(df, rank_col="Rank", target_col="Target")
    spearman = spearman_corr(df, pred_col="pred", target_col="Target")
    print(f"  {split_name}: Sharpe={sharpe:.4f}, Spearman={spearman:.4f}, "
          f"日均spread={daily_returns.mean():.6f}, 天数={len(daily_returns)}")


def train_lightgbm(df, feature_cols=None, params=None, n_rounds=800, use_early_stopping=False):
    if params is None:
        params = LGB_PARAMS.copy()
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    train_df, valid_df, test_df = split_time_series(df)
    train_df = train_df.dropna(subset=["Target"])
    valid_df = valid_df.dropna(subset=["Target"])
    test_df = test_df.dropna(subset=["Target"])

    X_train = train_df[feature_cols]
    y_train = train_df["Target"]
    X_valid = valid_df[feature_cols]
    y_valid = valid_df["Target"]

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    print(f"开始训练 LightGBM, {n_rounds} 轮...")
    callbacks = [
        lgb.log_evaluation(period=100),
    ]
    if use_early_stopping:
        callbacks.append(lgb.early_stopping(stopping_rounds=200))
        model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            valid_sets=[dtrain, dvalid],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
    else:
        model = lgb.train(
            params, dtrain,
            num_boost_round=n_rounds,
            callbacks=callbacks,
        )

    model_path = MODEL_DIR / "lgb_model.txt"
    model.save_model(str(model_path))
    print(f"模型已保存到: {model_path}")

    print("评估结果:")
    evaluate_model(model, valid_df, feature_cols, "验证集")
    evaluate_model(model, test_df, feature_cols, "测试集")

    importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importance(importance_type="gain"),
    }).sort_values("importance", ascending=False)
    print("\n特征重要性 Top 20:")
    print(importance.head(20).to_string(index=False))

    return model, importance


def load_model(model_name="lgb_model.txt"):
    model_path = MODEL_DIR / model_name
    model = lgb.Booster(model_file=str(model_path))
    return model
