from src.data.loader import load_merged
from src.data.preprocessor import preprocess
from src.features.build_features import build_all_features, get_feature_columns
from src.models.train_lgb import train_lightgbm


def main():
    print("=" * 60)
    print("JPX Tokyo Stock Exchange Prediction Pipeline")
    print("=" * 60)

    print("\n[1/5] Loading data (train + supplemental)...")
    df = load_merged(use_train=True, use_supplement=True)

    print("\n[2/5] Preprocessing...")
    df = preprocess(df)

    print("\n[3/5] Building features...")
    df = build_all_features(df, use_cache=True)
    feature_cols = get_feature_columns(df)
    print(f"Features: {len(feature_cols)}")

    print("\n[4/5] Training model (fixed 1000 rounds)...")
    model, importance = train_lightgbm(df, feature_cols=feature_cols, n_rounds=1000, use_early_stopping=False)

    print("\n[5/5] Done!")
    print(f"Total: {len(df)} rows, {df['Date'].min().date()} ~ {df['Date'].max().date()}")
    print(f"Features: {len(feature_cols)}")

    return model, importance, df


if __name__ == "__main__":
    model, importance, df = main()
