import numpy as np

from . import aggregate, normalise


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
        S = normalise.dimension_scores(normalise.minmax(Xp))
        scores[i] = aggregate.additive(S, w)
    return scores, rank_frequencies(scores)


def rank_frequencies(scores):
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    n_sys = scores.shape[1]
    return np.vstack([(ranks == r).mean(axis=0) for r in range(1, n_sys + 1)])
