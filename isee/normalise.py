import numpy as np
from scipy.stats import rankdata

from .data import DIMENSIONS, indicator_dimensions


def minmax(X, axis=0):
    X = np.asarray(X, float)
    lo = np.nanmin(X, axis=axis, keepdims=True)
    hi = np.nanmax(X, axis=axis, keepdims=True)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / rng


def zscore(X, axis=0):
    X = np.asarray(X, float)
    sd = X.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - X.mean(axis=axis, keepdims=True)) / sd


def percentile_rank(X, axis=0):
    X = np.asarray(X, float)
    n = X.shape[axis]
    if n <= 1:
        return np.zeros_like(X)
    R = np.apply_along_axis(lambda col: rankdata(col, method="average"),
                            axis, X)
    return (R - 1) / (n - 1)


def dimension_scores(X_norm):
    dims = np.array(indicator_dimensions())
    return np.column_stack(
        [np.nanmean(X_norm[:, dims == d], axis=1) for d in DIMENSIONS])
