import numpy as np
import pandas as pd
from src.config import TRAIN_DIR, SUPPLEMENT_DIR

FINANCIAL_COLS = [
    "Date", "SecuritiesCode", "DisclosedDate",
    "NetSales", "OperatingProfit", "Profit", "EarningsPerShare",
    "TotalAssets", "Equity", "EquityToAssetRatio",
    "ForecastProfit", "ForecastEarningsPerShare",
]

NUMERIC_COLS = [
    "NetSales", "OperatingProfit", "Profit", "EarningsPerShare",
    "TotalAssets", "Equity", "EquityToAssetRatio",
    "ForecastProfit", "ForecastEarningsPerShare",
]


def load_financials(use_train=True, use_supplement=False):
    dfs = []
    if use_train:
        df = pd.read_csv(TRAIN_DIR / "financials.csv", low_memory=False)
        dfs.append(df)
    if use_supplement:
        df = pd.read_csv(SUPPLEMENT_DIR / "financials.csv", low_memory=False)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    df = pd.concat(dfs, ignore_index=True)
    cols_avail = [c for c in FINANCIAL_COLS if c in df.columns]
    df = df[cols_avail].copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Date"] = pd.to_datetime(df["Date"])
    df["DisclosedDate"] = pd.to_datetime(df["DisclosedDate"], errors="coerce")
    df = df.dropna(subset=["SecuritiesCode"])
    return df


def _compute_ratios(df):
    df["OperatingMargin"] = df["OperatingProfit"] / df["NetSales"].replace(0, np.nan)
    df["ROE"] = df["Profit"] / df["Equity"].replace(0, np.nan)
    df["EPS_surprise"] = (df["EarningsPerShare"] - df["ForecastEarningsPerShare"]) / (
        df["ForecastEarningsPerShare"].abs().replace(0, np.nan)
    )
    return df


def build_financial_features(prices_df, use_train=True, use_supplement=False):
    fin = load_financials(use_train=use_train, use_supplement=use_supplement)
    if fin.empty:
        return prices_df

    fin = _compute_ratios(fin)

    merge_cols = [
        "SecuritiesCode", "DisclosedDate",
        "OperatingMargin", "ROE",
        "EPS_surprise", "EquityToAssetRatio",
    ]
    avail = [c for c in merge_cols if c in fin.columns]
    fin_merge = fin[avail].drop_duplicates(subset=["SecuritiesCode", "DisclosedDate"], keep="last")
    fin_merge = fin_merge.sort_values(["SecuritiesCode", "DisclosedDate"])

    prices_df = prices_df.copy()
    prices_df = prices_df.sort_values(["SecuritiesCode", "Date"])

    feature_cols = [c for c in avail if c not in ("SecuritiesCode", "DisclosedDate")]
    for col in feature_cols:
        prices_df[col] = np.nan

    stock_codes = prices_df["SecuritiesCode"].unique()
    fin_dict = {code: group for code, group in fin_merge.groupby("SecuritiesCode")}

    for code in stock_codes:
        if code not in fin_dict:
            continue
        f_data = fin_dict[code].copy()
        mask = prices_df["SecuritiesCode"] == code
        p_dates = prices_df.loc[mask, "Date"]

        f_data = f_data.sort_values("DisclosedDate")
        for col in feature_cols:
            if col not in f_data.columns:
                continue
            f_series = f_data.set_index("DisclosedDate")[col]
            f_series = f_series[~f_series.index.duplicated(keep="last")]
            combined_idx = f_series.index.union(p_dates.values).sort_values()
            result = f_series.reindex(combined_idx)
            result = result.ffill()
            result = result.reindex(p_dates.values)
            prices_df.loc[mask, col] = result.values

    for col in feature_cols:
        prices_df[col] = prices_df[col].replace([np.inf, -np.inf], np.nan)
        prices_df[col] = prices_df[col].clip(-1e4, 1e4)

    return prices_df
