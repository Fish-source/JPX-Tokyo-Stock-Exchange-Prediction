import numpy as np
import pandas as pd
from src.config import (
    MODEL_DIR, DATA_DIR, TRAIN_DIR, SUPPLEMENT_DIR, TEST_DIR,
)
from src.data.loader import load_stock_list
from src.data.preprocessor import preprocess
from src.features.price_features import build_price_features
from src.features.cross_section import build_cross_section_features
from src.features.financial_features import build_financial_features
from src.features.build_features import get_feature_columns
from src.models.ensemble import predict_with_models


def _merge_stock_list(df):
    stock_list = load_stock_list()
    merge_cols = ["SecuritiesCode", "33SectorCode", "33SectorName",
                  "17SectorCode", "17SectorName", "NewMarketSegment",
                  "MarketCapitalization", "Universe0"]
    stock_subset = stock_list[merge_cols].drop_duplicates(subset=["SecuritiesCode"])
    df = df.merge(stock_subset, on="SecuritiesCode", how="left")
    return df


def build_historical_buffer(df, max_window=65):
    buffer = {}
    for code, group in df.groupby("SecuritiesCode"):
        buffer[code] = group.sort_values("Date").tail(max_window).copy()
    return buffer


def compute_features_for_day(day_prices, buffer):
    codes = day_prices["SecuritiesCode"].unique()
    frames = []
    for code in codes:
        if code in buffer:
            combined = pd.concat([buffer[code], day_prices[day_prices["SecuritiesCode"] == code]],
                                 ignore_index=True)
        else:
            combined = day_prices[day_prices["SecuritiesCode"] == code].copy()
        frames.append(combined)
    df = pd.concat(frames, ignore_index=True)

    df = preprocess(df)
    df = _merge_stock_list(df)
    df = build_price_features(df)
    df = build_cross_section_features(df, use_train=True, use_supplement=True)
    df = build_financial_features(df, use_train=True, use_supplement=True)

    today = df["Date"].max()
    day_df = df[df["Date"] == today].copy()
    return day_df


def update_buffer(buffer, day_df, max_window=65):
    for code in day_df["SecuritiesCode"].unique():
        group = day_df[day_df["SecuritiesCode"] == code]
        if code in buffer:
            combined = pd.concat([buffer[code], group], ignore_index=True)
            buffer[code] = combined.sort_values("Date").tail(max_window).copy()
        else:
            buffer[code] = group.copy()
    return buffer


def load_ensemble_models():
    import lightgbm as lgb
    import xgboost as xgb

    models = []
    model_types = []
    weights = []

    lgb_path = MODEL_DIR / "lgb_model.txt"
    if lgb_path.exists():
        models.append(lgb.Booster(model_file=str(lgb_path)))
        model_types.append("lgb")
        weights.append(1.0)

    lgb_rank_path = MODEL_DIR / "lgb_rank_model.txt"
    if lgb_rank_path.exists():
        models.append(lgb.Booster(model_file=str(lgb_rank_path)))
        model_types.append("lgb")
        weights.append(1.0)

    xgb_path = MODEL_DIR / "xgb_model.json"
    if xgb_path.exists():
        bst = xgb.Booster()
        bst.load_model(str(xgb_path))
        models.append(bst)
        model_types.append("xgb")
        weights.append(1.0)

    return models, model_types, weights


def make_submission_iter_test(env, models, model_types, weights, feature_cols, initial_df=None, max_window=65):
    iter_test = env.iter_test()
    buffer = {}
    if initial_df is not None:
        initial_df = preprocess(initial_df)
        initial_df = _merge_stock_list(initial_df)
        buffer = build_historical_buffer(initial_df, max_window)

    for (prices, options, financials, trades,
         secondary_prices, sample_prediction) in iter_test:
        prices["Date"] = pd.to_datetime(prices["Date"])
        prices["Target"] = np.nan

        day_features = compute_features_for_day(prices, buffer)

        valid_features = [c for c in feature_cols if c in day_features.columns]
        day_features["pred"] = predict_with_models(
            models, day_features[valid_features], model_types, weights
        )

        pred_map = day_features.set_index("SecuritiesCode")["pred"]
        sample_prediction["Rank"] = sample_prediction["SecuritiesCode"].map(pred_map)
        sample_prediction["Rank"] = sample_prediction["Rank"].fillna(0)
        sample_prediction["Rank"] = sample_prediction["Rank"].rank(
            ascending=False, method="first"
        ).astype(int) - 1

        env.predict(sample_prediction)
        buffer = update_buffer(buffer, day_features, max_window)


def main():
    import sys
    sys.path.insert(0, str(DATA_DIR))
    from jpx_tokyo_market_prediction import make_env

    print("加载模型...")
    models, model_types, weights = load_ensemble_models()
    if not models:
        print("未找到训练好的模型，请先运行 run_pipeline.py")
        return
    print(f"已加载 {len(models)} 个模型: {model_types}")

    env = make_env()
    initial_df = pd.read_csv(TRAIN_DIR / "stock_prices.csv")
    initial_df["Date"] = pd.to_datetime(initial_df["Date"])

    sample = pd.read_csv(TEST_DIR / "stock_prices.csv", nrows=10)
    feature_cols_sample = get_feature_columns(sample)
    print(f"特征数: {len(feature_cols_sample)}")

    print("开始提交...")
    make_submission_iter_test(env, models, model_types, weights, feature_cols_sample, initial_df)
    print("提交完成!")


if __name__ == "__main__":
    main()
