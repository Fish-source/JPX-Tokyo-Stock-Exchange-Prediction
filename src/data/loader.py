import pandas as pd
from src.config import TRAIN_DIR, DATA_DIR


def load_stock_prices(use_train=True, use_supplement=False):
    dfs = []
    if use_train:
        df = pd.read_csv(TRAIN_DIR / "stock_prices.csv")
        dfs.append(df)
    if use_supplement:
        from src.config import SUPPLEMENT_DIR
        df = pd.read_csv(SUPPLEMENT_DIR / "stock_prices.csv")
        dfs.append(df)
    if not dfs:
        raise ValueError("至少需要加载一个数据源")
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    return df


def load_stock_list():
    df = pd.read_csv(DATA_DIR / "stock_list.csv")
    return df


def load_merged(use_train=True, use_supplement=False):
    prices = load_stock_prices(use_train=use_train, use_supplement=use_supplement)
    stock_list = load_stock_list()
    merge_cols = ["SecuritiesCode", "33SectorCode", "33SectorName",
                  "17SectorCode", "17SectorName", "NewMarketSegment",
                  "MarketCapitalization", "Universe0"]
    stock_subset = stock_list[merge_cols].copy()
    df = prices.merge(stock_subset, on="SecuritiesCode", how="left")
    return df
