import numpy as np
import pandas as pd


def calc_return_rank(df):
    if "Return_1d" not in df.columns:
        return df
    df["ReturnRank"] = df.groupby("Date")["Return_1d"].rank(
        ascending=False, method="first"
    )
    n_stocks = df.groupby("Date")["SecuritiesCode"].transform("count")
    df["ReturnRankPct"] = df["ReturnRank"] / n_stocks
    return df


def calc_sector_return_mean(df):
    if "33SectorName" not in df.columns:
        return df
    for w in [1, 5, 20]:
        ret_col = f"Return_{w}d" if w > 1 else "Return_1d"
        if ret_col not in df.columns:
            continue
        df[f"SectorReturn_{w}d"] = df.groupby(["Date", "33SectorName"])[ret_col].transform("mean")
        df[f"SectorExcess_{w}d"] = df[ret_col] - df[f"SectorReturn_{w}d"]
    return df


def calc_sector_rank(df):
    if "33SectorName" not in df.columns:
        return df
    if "Return_1d" not in df.columns:
        return df
    df["SectorRank"] = df.groupby(["Date", "33SectorName"])["Return_1d"].rank(
        ascending=False, method="first"
    )
    sector_size = df.groupby(["Date", "33SectorName"])["SecuritiesCode"].transform("count")
    df["SectorRankPct"] = df["SectorRank"] / sector_size
    return df


def calc_market_return(df):
    if "Return_1d" not in df.columns:
        return df
    df["MarketReturn_1d"] = df.groupby("Date")["Return_1d"].transform("mean")
    df["ExcessReturn_1d"] = df["Return_1d"] - df["MarketReturn_1d"]
    return df


def calc_volatility_rank(df):
    if "Volatility_20d" not in df.columns:
        return df
    df["VolatilityRank"] = df.groupby("Date")["Volatility_20d"].rank(
        ascending=False, method="first"
    )
    n_stocks = df.groupby("Date")["SecuritiesCode"].transform("count")
    df["VolatilityRankPct"] = df["VolatilityRank"] / n_stocks
    return df


def calc_volume_rank(df):
    df["VolumeRank"] = df.groupby("Date")["Volume"].rank(
        ascending=True, method="first"
    )
    n_stocks = df.groupby("Date")["SecuritiesCode"].transform("count")
    df["VolumeRankPct"] = df["VolumeRank"] / n_stocks
    return df


def calc_target_lag_rank(df):
    if "TargetLag_1d" not in df.columns:
        return df
    df["TargetLag1dRank"] = df.groupby("Date")["TargetLag_1d"].rank(
        ascending=False, method="first"
    )
    n_stocks = df.groupby("Date")["SecuritiesCode"].transform("count")
    df["TargetLag1dRankPct"] = df["TargetLag1dRank"] / n_stocks
    return df


def calc_calendar_features(df):
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["Month"] = df["Date"].dt.month
    df["DayOfMonth"] = df["Date"].dt.day
    df["IsMonthStart"] = (df["DayOfMonth"] <= 5).astype(int)
    df["IsMonthEnd"] = (df["DayOfMonth"] >= 25).astype(int)
    return df


def build_cross_section_features(df):
    df = calc_return_rank(df)
    df = calc_sector_return_mean(df)
    df = calc_sector_rank(df)
    df = calc_market_return(df)
    df = calc_volatility_rank(df)
    df = calc_volume_rank(df)
    df = calc_target_lag_rank(df)
    df = calc_calendar_features(df)
    return df
