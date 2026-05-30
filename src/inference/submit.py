import numpy as np
import pandas as pd
from src.data.loader import load_stock_list
from src.data.preprocessor import preprocess
from src.features.price_features import build_price_features
from src.features.cross_section import build_cross_section_features
from src.features.build_features import get_feature_columns


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
    df = build_cross_section_features(df)

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


def make_submission_iter_test(env, model, feature_cols, initial_df=None, max_window=65):
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
        day_features["pred"] = model.predict(day_features[valid_features])

        pred_map = day_features.set_index("SecuritiesCode")["pred"]
        sample_prediction["Rank"] = sample_prediction["SecuritiesCode"].map(pred_map)
        sample_prediction["Rank"] = sample_prediction["Rank"].fillna(0)
        sample_prediction["Rank"] = sample_prediction["Rank"].rank(
            ascending=False, method="first"
        ).astype(int) - 1

        env.predict(sample_prediction)

        buffer = update_buffer(buffer, day_features, max_window)
