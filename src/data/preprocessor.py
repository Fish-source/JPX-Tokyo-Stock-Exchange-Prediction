import numpy as np
import pandas as pd


def adjust_prices(df):
    df = df.copy()
    df = df.sort_values(["SecuritiesCode", "Date"])
    df["AdjustmentFactor"] = df["AdjustmentFactor"].fillna(1.0)
    df["CumAdjustmentFactor"] = df.groupby("SecuritiesCode")["AdjustmentFactor"] \
        .transform(lambda x: x[::-1].cumprod()[::-1])
    for col in ["Open", "High", "Low", "Close"]:
        df[f"Adjusted{col}"] = df[col] / df["CumAdjustmentFactor"]
    return df


def fill_missing_prices(df):
    df = df.copy()
    price_cols = ["AdjustedOpen", "AdjustedHigh", "AdjustedLow", "AdjustedClose"]
    for col in price_cols:
        df[col] = df.groupby("SecuritiesCode")[col].transform(lambda x: x.ffill())
    df["Volume"] = df.groupby("SecuritiesCode")["Volume"].transform(lambda x: x.fillna(0))
    return df


def clean_target(df):
    df = df.copy()
    df["Target"] = df["Target"].replace([np.inf, -np.inf], np.nan)
    return df


def preprocess(df):
    df = adjust_prices(df)
    df = fill_missing_prices(df)
    df = clean_target(df)
    df["ExpectedDividend"] = df["ExpectedDividend"].fillna(0)
    return df
