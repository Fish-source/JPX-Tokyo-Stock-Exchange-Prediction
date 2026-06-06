import numpy as np
import pandas as pd


def fill_missing_prices(df):
    df = df.copy()
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df.groupby("SecuritiesCode")[col].transform(
            lambda x: x.ffill().bfill()
        )
    return df


def zscore_prices(df):
    df = df.copy()
    zscore_cols = ["Open", "High", "Low", "Close", "Volume"]
    for col in zscore_cols:
        df[col + "_z"] = df.groupby("SecuritiesCode")[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
    df["Daily_Range_z"] = df["Close_z"] - df["Open_z"]
    df["Mean_z"] = (df["High_z"] + df["Low_z"]) / 2
    return df


def preprocess(df):
    df = fill_missing_prices(df)
    df = zscore_prices(df)
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    df["SupervisionFlag"] = df["SupervisionFlag"].astype(int)
    df["Target"] = df["Target"].replace([np.inf, -np.inf], np.nan)
    return df
