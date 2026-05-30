import numpy as np
import pandas as pd


def calc_spread_return_per_day(df, rank_col="Rank", target_col="Target"):
    df = df.copy()
    n_stocks = len(df)
    top_n = min(200, n_stocks // 10 * 2)
    bottom_n = top_n
    df["weight"] = np.zeros(n_stocks)
    top_idx = df[rank_col].nsmallest(top_n).index
    bottom_idx = df[rank_col].nlargest(bottom_n).index
    top_weights = 2.0 * (top_n - df.loc[top_idx, rank_col]) / (top_n * (top_n + 1))
    bottom_weights = 2.0 * (bottom_n - (n_stocks - 1 - df.loc[bottom_idx, rank_col])) / (bottom_n * (bottom_n + 1))
    df.loc[top_idx, "weight"] = top_weights.values
    df.loc[bottom_idx, "weight"] = -bottom_weights.values
    spread_return = (df["weight"] * df[target_col]).sum()
    return spread_return


def calc_spread_return_sharpe(df, rank_col="Rank", target_col="Target"):
    daily_returns = df.groupby("Date").apply(
        lambda x: calc_spread_return_per_day(x, rank_col, target_col)
    )
    if len(daily_returns) < 2:
        return 0.0, pd.Series(dtype=float)
    sharpe = daily_returns.mean() / daily_returns.std()
    return sharpe, daily_returns


def rank_prediction(df, pred_col="pred"):
    df = df.copy()
    df["Rank"] = df.groupby("Date")[pred_col].rank(
        ascending=False, method="first"
    ).astype(int) - 1
    return df


def spearman_corr(df, pred_col="pred", target_col="Target"):
    from scipy.stats import spearmanr
    correlations = []
    for date, group in df.groupby("Date"):
        if len(group) < 10:
            continue
        valid = group[[pred_col, target_col]].dropna()
        if len(valid) < 10:
            continue
        corr, _ = spearmanr(valid[pred_col], valid[target_col])
        correlations.append(corr)
    return np.mean(correlations) if correlations else 0.0
