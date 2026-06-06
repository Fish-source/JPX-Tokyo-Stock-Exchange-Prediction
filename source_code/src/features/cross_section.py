import numpy as np
import pandas as pd
from src.config import BASE_FEATURES, CS_RANK_FEATURES


def build_cross_section_features(df):
    rank_sources = {
        "Return_1d_rank": "Return_1d",
        "Return_5d_rank": "Return_5d",
        "Return_10d_rank": "Return_10d",
        "Return_20d_rank": "Return_20d",
        "Volatility_20d_rank": "Volatility_20d",
        "TargetLag_1d_rank": "TargetLag_1d",
        "Close_z_rank": "Close_z",
        "Volume_z_rank": "Volume_z",
        "IntradayRange_rank": "IntradayRange",
    }
    for rank_col, source_col in rank_sources.items():
        if source_col in df.columns:
            df[rank_col] = df.groupby("Date")[source_col].rank(
                ascending=True, method="average", pct=True
            )
    return df
