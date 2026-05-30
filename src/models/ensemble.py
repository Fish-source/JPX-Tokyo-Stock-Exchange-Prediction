import numpy as np
import pandas as pd


def simple_average(models, X, weights=None):
    preds = []
    for model in models:
        pred = model.predict(X)
        preds.append(pred)
    preds = np.array(preds)
    if weights is None:
        return preds.mean(axis=0)
    weights = np.array(weights)
    weights = weights / weights.sum()
    return (preds * weights[:, None]).sum(axis=0)
