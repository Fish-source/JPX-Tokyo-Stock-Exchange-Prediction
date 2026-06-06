import numpy as np
import pandas as pd
from src.config import MODEL_DIR
from src.evaluation.metrics import calc_spread_return_sharpe, rank_prediction, spearman_corr


def predict_with_lgb_seeds(models, X):
    preds = np.array([m.predict(X) for m in models])
    return preds.mean(axis=0)


def predict_with_models(lgb_models, xgb_model, ridge_model, ridge_scaler, ridge_cols,
                        X_lgb, X_xgb, X_ridge):
    preds = []

    if lgb_models:
        lgb_pred = predict_with_lgb_seeds(lgb_models, X_lgb)
        preds.append(lgb_pred)

    if xgb_model is not None:
        import xgboost as xgb
        xgb_pred = xgb_model.predict(xgb.DMatrix(X_xgb.fillna(0)))
        preds.append(xgb_pred)

    if ridge_model is not None:
        X = X_ridge.fillna(0)
        if ridge_scaler is not None:
            X = ridge_scaler.transform(X)
        ridge_pred = ridge_model.predict(X)
        preds.append(ridge_pred)

    preds = np.array(preds)
    return preds.mean(axis=0)


def optimize_weights(lgb_models, xgb_model, ridge_model, ridge_scaler, ridge_cols,
                     valid_df, lgb_feat, xgb_feat):
    all_preds = []

    if lgb_models:
        lgb_pred = predict_with_lgb_seeds(lgb_models, valid_df[lgb_feat])
        all_preds.append(("lgb", lgb_pred))

    if xgb_model is not None:
        import xgboost as xgb
        xgb_pred = xgb_model.predict(xgb.DMatrix(valid_df[xgb_feat].fillna(0)))
        all_preds.append(("xgb", xgb_pred))

    if ridge_model is not None:
        X = valid_df[ridge_cols].fillna(0)
        if ridge_scaler is not None:
            X = ridge_scaler.transform(X)
        ridge_pred = ridge_model.predict(X)
        all_preds.append(("ridge", ridge_pred))

    n = len(all_preds)
    if n == 0:
        return []

    pred_array = np.array([p[1] for p in all_preds])

    best_sharpe = -np.inf
    best_weights = [1.0 / n] * n

    if n == 2:
        grid = np.linspace(0.1, 0.9, 9)
        for w1 in grid:
            w = np.array([w1, 1 - w1])
            blend = (pred_array * w[:, None]).sum(axis=0)
            valid_df_copy = valid_df.copy()
            valid_df_copy["pred"] = blend
            valid_df_copy = rank_prediction(valid_df_copy, pred_col="pred")
            sharpe, _ = calc_spread_return_sharpe(valid_df_copy, rank_col="Rank", target_col="Target")
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_weights = w.tolist()
    elif n == 3:
        grid = np.linspace(0.05, 0.8, 16)
        for w1 in grid:
            for w2 in grid:
                w3 = 1.0 - w1 - w2
                if w3 < 0.05:
                    continue
                w = np.array([w1, w2, w3])
                blend = (pred_array * w[:, None]).sum(axis=0)
                valid_df_copy = valid_df.copy()
                valid_df_copy["pred"] = blend
                valid_df_copy = rank_prediction(valid_df_copy, pred_col="pred")
                sharpe, _ = calc_spread_return_sharpe(valid_df_copy, rank_col="Rank", target_col="Target")
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_weights = w.tolist()

    names = [p[0] for p in all_preds]
    print(f"Ensemble weights: {dict(zip(names, best_weights))}, valid Sharpe: {best_sharpe:.4f}")
    return best_weights


def evaluate_ensemble(lgb_models, xgb_model, ridge_model, ridge_scaler, ridge_cols,
                      valid_df, test_df, lgb_feat, xgb_feat, weights=None):
    if weights is None:
        weights = optimize_weights(lgb_models, xgb_model, ridge_model, ridge_scaler,
                                   ridge_cols, valid_df, lgb_feat, xgb_feat)

    for split_name, split_df in [("Valid", valid_df), ("Test", test_df)]:
        split_df = split_df.dropna(subset=["Target"]).copy()
        if len(split_df) == 0:
            continue

        all_preds = []
        if lgb_models:
            all_preds.append(predict_with_lgb_seeds(lgb_models, split_df[lgb_feat]))
        if xgb_model is not None:
            import xgboost as xgb
            all_preds.append(xgb_model.predict(xgb.DMatrix(split_df[xgb_feat].fillna(0))))
        if ridge_model is not None:
            X = split_df[ridge_cols].fillna(0)
            if ridge_scaler is not None:
                X = ridge_scaler.transform(X)
            all_preds.append(ridge_model.predict(X))

        pred_array = np.array(all_preds)
        w = np.array(weights, dtype=float)
        w = w / w.sum()
        blend = (pred_array * w[:, None]).sum(axis=0)

        split_df["pred"] = blend
        split_df = rank_prediction(split_df, pred_col="pred")
        sharpe, _ = calc_spread_return_sharpe(split_df, rank_col="Rank", target_col="Target")
        sp = spearman_corr(split_df, pred_col="pred", target_col="Target")
        print(f"  Ensemble {split_name}: Sharpe={sharpe:.4f}, Spearman={sp:.4f}")

    return weights
