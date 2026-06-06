import numpy as np
import pandas as pd
from src.config import FEATURE_DIR, FEATURE_CACHE_NAME, BASE_FEATURES, CS_RANK_FEATURES
from src.features.price_features import build_price_features
from src.features.cross_section import build_cross_section_features


def build_all_features(df, use_cache=True):
    cache_path = FEATURE_DIR / FEATURE_CACHE_NAME
    if use_cache and cache_path.exists():
        print(f"Loading features from cache: {cache_path}")
        return pd.read_parquet(cache_path)

    df = build_price_features(df)
    df = build_cross_section_features(df)

    df = _clean_features(df)

    if use_cache:
        print(f"Caching features to: {cache_path}")
        df.to_parquet(cache_path, index=False)

    return df


def _clean_features(df):
    all_feat = BASE_FEATURES + CS_RANK_FEATURES
    for col in all_feat:
        if col not in df.columns:
            continue
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        if df[col].dtype in [np.float64, np.float32, float]:
            q01 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            if pd.notna(q01) and pd.notna(q99):
                df[col] = df[col].clip(q01, q99)
    return df


def get_base_features():
    return [f for f in BASE_FEATURES]


def get_cs_features():
    return [f for f in CS_RANK_FEATURES]
