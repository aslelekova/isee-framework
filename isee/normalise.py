import numpy as np

from .data import DIMENSIONS, indicator_dimensions


def minmax(X, axis=0):
    X = np.asarray(X, float)
    lo, hi = X.min(axis=axis, keepdims=True), X.max(axis=axis, keepdims=True)
    rng = np.where(hi - lo == 0, 1.0, hi - lo)
    return (X - lo) / rng


def zscore(X, axis=0):
    X = np.asarray(X, float)
    sd = X.std(axis=axis, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - X.mean(axis=axis, keepdims=True)) / sd


def percentile_rank(X, axis=0):
    X = np.asarray(X, float)
    order = X.argsort(axis=axis).argsort(axis=axis)
    n = X.shape[axis]
    return order / (n - 1) if n > 1 else np.zeros_like(X)


def dimension_scores(X_norm):
    dims = np.array(indicator_dimensions())
    return np.column_stack(
        [X_norm[:, dims == d].mean(axis=1) for d in DIMENSIONS])
