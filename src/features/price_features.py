import numpy as np
import pandas as pd
from src.config import MOMENTUM_WINDOWS, VOL_WINDOWS

INF_CLIP = 1e6


def calc_returns(df):
    for w in MOMENTUM_WINDOWS:
        col = f"Return_{w}d"
        df[col] = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
            lambda x: x / x.shift(w) - 1
        )
        df[col] = df[col].clip(-INF_CLIP, INF_CLIP)
    return df


def calc_volatility(df):
    df["DailyReturn"] = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
        lambda x: x.pct_change()
    )
    df["DailyReturn"] = df["DailyReturn"].clip(-INF_CLIP, INF_CLIP)
    for w in VOL_WINDOWS:
        col = f"Volatility_{w}d"
        df[col] = df.groupby("SecuritiesCode")["DailyReturn"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 1)).std()
        )
    return df


def calc_volume_features(df):
    for w in [5, 10, 20]:
        rolling_mean = df.groupby("SecuritiesCode")["Volume"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 1)).mean()
        )
        df[f"VolumeRatio_{w}d"] = df["Volume"] / (rolling_mean + 1)
        df[f"VolumeRatio_{w}d"] = df[f"VolumeRatio_{w}d"].clip(0, INF_CLIP)
    df["VolumeChange"] = df.groupby("SecuritiesCode")["Volume"].transform(
        lambda x: x.pct_change()
    )
    df["VolumeChange"] = df["VolumeChange"].clip(-INF_CLIP, INF_CLIP)
    return df


def calc_intraday_features(df):
    close = df["AdjustedClose"].replace(0, np.nan)
    hl_range = df["AdjustedHigh"] - df["AdjustedLow"]
    df["IntradayRange"] = (hl_range / close).fillna(0)
    max_oc = df[["AdjustedOpen", "AdjustedClose"]].max(axis=1)
    min_oc = df[["AdjustedOpen", "AdjustedClose"]].min(axis=1)
    df["UpperShadow"] = ((df["AdjustedHigh"] - max_oc) / close).fillna(0)
    df["LowerShadow"] = ((min_oc - df["AdjustedLow"]) / close).fillna(0)
    df["BodyRatio"] = (abs(df["AdjustedClose"] - df["AdjustedOpen"]) / (hl_range + 1e-8)).fillna(0)
    df["OpenCloseReturn"] = ((df["AdjustedClose"] - df["AdjustedOpen"]) / df["AdjustedOpen"]).clip(-INF_CLIP, INF_CLIP).fillna(0)
    return df


def calc_drawdown(df):
    for w in [20, 40, 60]:
        rolling_max = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
            lambda x: x.rolling(w, min_periods=max(w // 2, 1)).max()
        )
        df[f"Drawdown_{w}d"] = (df["AdjustedClose"] / rolling_max - 1).fillna(0)
        df[f"Drawdown_{w}d"] = df[f"Drawdown_{w}d"].clip(-1, 0)
    return df


def calc_rsi(df, period=14):
    delta = df.groupby("SecuritiesCode")["AdjustedClose"].transform(lambda x: x.diff())
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(df["SecuritiesCode"]).transform(
        lambda x: x.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    )
    avg_loss = loss.groupby(df["SecuritiesCode"]).transform(
        lambda x: x.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    )
    rs = avg_gain / (avg_loss + 1e-8)
    df[f"RSI_{period}"] = (100 - 100 / (1 + rs)).fillna(50)
    return df


def calc_macd(df, fast=12, slow=26, signal=9):
    ema_fast = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
        lambda x: x.ewm(span=fast, adjust=False).mean()
    )
    ema_slow = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
        lambda x: x.ewm(span=slow, adjust=False).mean()
    )
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df.groupby("SecuritiesCode")["MACD"].transform(
        lambda x: x.ewm(span=signal, adjust=False).mean()
    )
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    return df


def calc_bollinger_position(df, window=20):
    rolling_mean = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
        lambda x: x.rolling(window, min_periods=window).mean()
    )
    rolling_std = df.groupby("SecuritiesCode")["AdjustedClose"].transform(
        lambda x: x.rolling(window, min_periods=window).std()
    )
    df["BollingerPos"] = ((df["AdjustedClose"] - rolling_mean) / (2 * rolling_std + 1e-8)).fillna(0)
    return df


def calc_dividend_yield(df):
    df["DividendYield"] = (df["ExpectedDividend"] / (df["AdjustedClose"] + 1e-8)).fillna(0)
    return df


def calc_lagged_target(df):
    for w in [1, 2, 3, 5, 10, 20]:
        df[f"TargetLag_{w}d"] = df.groupby("SecuritiesCode")["Target"].transform(
            lambda x: x.shift(w)
        )
    df["TargetRollingMean_5d"] = df.groupby("SecuritiesCode")["Target"].transform(
        lambda x: x.shift(1).rolling(5, min_periods=1).mean()
    )
    df["TargetRollingMean_20d"] = df.groupby("SecuritiesCode")["Target"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).mean()
    )
    df["TargetRollingStd_20d"] = df.groupby("SecuritiesCode")["Target"].transform(
        lambda x: x.shift(1).rolling(20, min_periods=5).std()
    )
    return df


def calc_momentum_reversal(df):
    df["MomentumReversal_5d"] = df["Return_5d"] - df["Return_1d"]
    df["MomentumReversal_20d"] = df["Return_1d"] - df["Return_20d"]
    df["MomentumReversal_60d"] = df["Return_1d"] - df["Return_60d"]
    return df


def build_price_features(df):
    df = calc_returns(df)
    df = calc_volatility(df)
    df = calc_volume_features(df)
    df = calc_intraday_features(df)
    df = calc_drawdown(df)
    df = calc_rsi(df)
    df = calc_macd(df)
    df = calc_bollinger_position(df)
    df = calc_dividend_yield(df)
    df = calc_lagged_target(df)
    df = calc_momentum_reversal(df)
    df = df.drop(columns=["DailyReturn"], errors="ignore")
    return df
