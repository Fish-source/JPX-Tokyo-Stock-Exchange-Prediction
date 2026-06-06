import numpy as np
import pandas as pd
from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_base_features, get_cs_features
from src.models.train_lgb import train_stacking


def main():
    print("=" * 60)
    print("JPX Tokyo Stock Exchange Prediction Pipeline v6")
    print("Stacking: Ridge + LightGBM on residuals")
    print("=" * 60)

    print("\n[1/4] Loading data (all sources including secondary)...")
    df = load_merged(use_train=True, use_supplement=True, use_secondary=True)

    print("\n[2/4] Preprocessing + z-scoring...")
    df = preprocess(df)

    print("\n[3/4] Building features...")
    df = build_all_features(df, use_cache=True)
    base_feats = get_base_features()
    cs_feats = get_cs_features()
    base_avail = [f for f in base_feats if f in df.columns]
    cs_avail = [f for f in cs_feats if f in df.columns]
    print(f"Base features: {len(base_avail)}, CS rank features: {len(cs_avail)}")

    print("\n[4/4] Training stacking model...")
    results = train_stacking(df, base_feats=base_avail, cs_feats=cs_avail)

    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"  Date range: {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"  Total rows: {len(df):,}")
    print(f"  Base features: {len(results['base_feats'])}")
    print(f"  CS features: {len(results['cs_feats'])}")
    print("=" * 60)

    return results


if __name__ == "__main__":
    results = main()
