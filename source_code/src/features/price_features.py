import numpy as np
import pandas as pd
from src.config import RETURN_WINDOWS

INF_CLIP = 1e6


def calc_returns(df):
    for w in RETURN_WINDOWS:
        col = f"Return_{w}d"
        df[col] = df.groupby("SecuritiesCode")["Close"].transform(
            lambda x: x / x.shift(w) - 1
        )
        df[col] = df[col].clip(-INF_CLIP, INF_CLIP)
    return df


def calc_volatility(df):
    df["DailyReturn"] = df.groupby("SecuritiesCode")["Close"].transform(
        lambda x: x.pct_change()
    )
    df["DailyReturn"] = df["DailyReturn"].clip(-INF_CLIP, INF_CLIP)
    df["Volatility_20d"] = df.groupby("SecuritiesCode")["DailyReturn"].transform(
        lambda x: x.rolling(20, min_periods=5).std()
    )
    return df


def calc_intraday_features(df):
    close = df["Close"].replace(0, np.nan)
    df["IntradayRange"] = ((df["High"] - df["Low"]) / close).fillna(0)
    return df


def calc_lagged_target(df):
    df["TargetLag_1d"] = df.groupby("SecuritiesCode")["Target"].transform(
        lambda x: x.shift(1)
    )
    return df


def build_price_features(df):
    df = calc_returns(df)
    df = calc_volatility(df)
    df = calc_intraday_features(df)
    df = calc_lagged_target(df)
    df = df.drop(columns=["DailyReturn"], errors="ignore")
    return df
