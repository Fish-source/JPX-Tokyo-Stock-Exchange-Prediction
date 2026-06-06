import pandas as pd
from src.config import TRAIN_DIR, DATA_DIR, SUPPLEMENT_DIR


def load_stock_prices(use_train=True, use_supplement=False, use_secondary=False):
    dfs = []
    if use_train:
        dfs.append(pd.read_csv(TRAIN_DIR / "stock_prices.csv"))
    if use_supplement:
        dfs.append(pd.read_csv(SUPPLEMENT_DIR / "stock_prices.csv"))
    if use_secondary:
        sec_train = TRAIN_DIR / "secondary_stock_prices.csv"
        sec_supp = SUPPLEMENT_DIR / "secondary_stock_prices.csv"
        if sec_train.exists():
            dfs.append(pd.read_csv(sec_train))
        if sec_supp.exists():
            dfs.append(pd.read_csv(sec_supp))
    if not dfs:
        raise ValueError("At least one data source required")
    df = pd.concat(dfs, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(["Date", "SecuritiesCode"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['SecuritiesCode'].nunique()} stocks, "
          f"{df['Date'].nunique()} days")
    return df


def load_stock_list():
    return pd.read_csv(DATA_DIR / "stock_list.csv")


def load_merged(use_train=True, use_supplement=False, use_secondary=False):
    prices = load_stock_prices(
        use_train=use_train,
        use_supplement=use_supplement,
        use_secondary=use_secondary,
    )
    stock_list = load_stock_list()
    merge_cols = [
        "SecuritiesCode", "33SectorCode", "33SectorName",
        "17SectorCode", "17SectorName", "NewMarketSegment",
        "MarketCapitalization", "Universe0",
    ]
    stock_subset = stock_list[merge_cols].copy()
    df = prices.merge(stock_subset, on="SecuritiesCode", how="left")
    return df
