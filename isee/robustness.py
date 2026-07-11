import numpy as np
from scipy.stats import rankdata

from . import aggregate, normalise
from .data import INDICATOR_DIMS, DIMENSIONS, build_indicators, perturb_primitives


def weight_mc(S, n=10_000, seed=42):
    rng = np.random.default_rng(seed)
    W = rng.dirichlet(np.ones(S.shape[1]), size=n)
    scores = np.einsum("nk,sk->ns", W, S * aggregate.SIGNS)
    return scores, rank_frequencies(scores)


def data_mc(ms, w, n=5_000, noise=0.10, seed=42):
    rng = np.random.default_rng(seed)
    scores = np.empty((n, len(ms)))
    for i in range(n):
        Xp = build_indicators(perturb_primitives(ms.P, ms.countries, rng,
                                                 noise))
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
            [np.nanmean(Xn[:, kept_dims == d], axis=1) for d in DIMENSIONS])
        sc = aggregate.additive(S, w)
        out.append({
            "dropped": drop,
            "scores": sc,
            "ranks": aggregate.ranks(sc),
            "max_rank_shift": int(np.abs(aggregate.ranks(sc) - base_rank).max()),
        })
    return out


def spearman_matrix(X):
    n = X.shape[1]
    C = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            m = ~(np.isnan(X[:, i]) | np.isnan(X[:, j]))
            a = rankdata(X[m, i], method="average")
            b = rankdata(X[m, j], method="average")
            C[i, j] = C[j, i] = np.corrcoef(a, b)[0, 1]
    return C
