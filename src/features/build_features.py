import numpy as np
import pandas as pd
from src.config import FEATURE_DIR, FEATURE_CACHE_NAME
from src.features.price_features import build_price_features
from src.features.cross_section import build_cross_section_features


def build_all_features(df, use_cache=True):
    cache_path = FEATURE_DIR / FEATURE_CACHE_NAME
    if use_cache and cache_path.exists():
        print(f"从缓存加载特征: {cache_path}")
        return pd.read_parquet(cache_path)

    df = build_price_features(df)
    df = build_cross_section_features(df)

    df = _clean_features(df)

    if use_cache:
        print(f"缓存特征到: {cache_path}")
        df.to_parquet(cache_path, index=False)

    return df


def _clean_features(df):
    feature_cols = get_feature_columns(df)
    for col in feature_cols:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].dtype in [np.float64, np.float32, float]:
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            if pd.notna(q01) and pd.notna(q99):
                df[col] = df[col].clip(q01, q99)
    return df


def get_feature_columns(df):
    exclude_cols = {
        "RowId", "Date", "SecuritiesCode", "Open", "High", "Low", "Close",
        "Volume", "AdjustmentFactor", "ExpectedDividend", "SupervisionFlag",
        "Target", "CumAdjustmentFactor", "AdjustedOpen", "AdjustedHigh",
        "AdjustedLow", "AdjustedClose",
        "33SectorCode", "33SectorName", "17SectorCode", "17SectorName",
        "NewMarketSegment", "MarketCapitalization", "Universe0",
    }
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols
