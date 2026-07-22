import numpy as np

from . import aggregate, normalise
from .data import INDICATOR_DIMS, INDICATOR_UPPER, DIMENSIONS


def weight_mc(S, n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(S.shape[1]), size=n)
    scores = np.einsum("nk,sk->ns", W, S * aggregate.SIGNS)
    return scores, rank_frequencies(scores)


def data_mc(X_raw, w, n=5_000, noise=0.10, seed=42):
    rng = np.random.default_rng(seed)
    n_sys = X_raw.shape[0]
    scores = np.empty((n, n_sys))
    for i in range(n):
        Xp = X_raw * rng.uniform(1 - noise, 1 + noise, size=X_raw.shape)
        Xp = np.clip(Xp, 0.0, INDICATOR_UPPER)
        S = normalise.dimension_scores(normalise.minmax(Xp))
        scores[i] = aggregate.additive(S, w)
    return scores, rank_frequencies(scores)


def rank_frequencies(scores):
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    n_sys = scores.shape[1]
    return np.vstack([(ranks == r).mean(axis=0) for r in range(1, n_sys + 1)])


def leave_one_indicator_out(X_raw, w):
    dims = np.array(INDICATOR_DIMS)
    base = aggregate.additive(
        normalise.dimension_scores(normalise.minmax(X_raw)), w)
    base_rank = aggregate.ranks(base)
    out = []
    for drop in range(X_raw.shape[1]):
        keep = np.arange(X_raw.shape[1]) != drop
        Xn = normalise.minmax(X_raw[:, keep])
        kept_dims = dims[keep]
        S = np.column_stack(
            [Xn[:, kept_dims == d].mean(axis=1) for d in DIMENSIONS])
        sc = aggregate.additive(S, w)
        out.append({
            "dropped": drop,
            "scores": sc,
            "ranks": aggregate.ranks(sc),
            "max_rank_shift": int(np.abs(aggregate.ranks(sc) - base_rank).max()),
        })
    return out


def spearman_matrix(X):
    R = np.apply_along_axis(
        lambda col: col.argsort().argsort().astype(float), 0, X)
    return np.corrcoef(R.T)
