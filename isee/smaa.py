import numpy as np

from . import aggregate, normalise

FLOOR = 0.1


def power_mean(S, w, rho):
    B = FLOOR + (1.0 - FLOOR) * aggregate.benefit_oriented(S)
    w = np.asarray(w, float)
    if abs(rho) < 1e-9:
        return np.exp(np.log(B) @ w)
    return (np.power(B, rho) @ w) ** (1.0 / rho)


def compensability_sweep(S, w, rho_grid=None):
    if rho_grid is None:
        rho_grid = np.arange(-2.0, 2.0 + 1e-9, 0.05)
    scores = np.vstack([power_mean(S, w, r) for r in rho_grid])
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return np.asarray(rho_grid), scores, ranks


def sample(X_raw, n=20_000, rho_range=(-2.0, 2.0), noise=0.10, seed=42,
           upper=None):
    rng = np.random.default_rng(seed)
    n_sys, n_ind = X_raw.shape
    W = rng.dirichlet(np.ones(5), size=n)
    rhos = rng.uniform(*rho_range, size=n)
    scores = np.empty((n, n_sys))
    for i in range(n):
        Xp = X_raw
        if noise > 0:
            Xp = X_raw * rng.uniform(1 - noise, 1 + noise, size=X_raw.shape)
            if upper is not None:
                Xp = np.clip(Xp, 0.0, upper)
        S = normalise.dimension_scores(normalise.minmax(Xp))
        scores[i] = power_mean(S, W[i], rhos[i])
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return {"weights": W, "rhos": rhos, "scores": scores, "ranks": ranks}


def rank_acceptability(ranks, n_sys=None):
    n_sys = n_sys or ranks.shape[1]
    return np.vstack([(ranks == r).mean(axis=0) for r in range(1, n_sys + 1)])


def pairwise_probability(scores):
    n_sys = scores.shape[1]
    P = np.zeros((n_sys, n_sys))
    for i in range(n_sys):
        for j in range(n_sys):
            if i != j:
                P[i, j] = (scores[:, i] > scores[:, j]).mean()
    return P


def central_weights(sample_result, rank=1):
    ranks, W = sample_result["ranks"], sample_result["weights"]
    out = []
    for j in range(ranks.shape[1]):
        m = ranks[:, j] == rank
        out.append(W[m].mean(axis=0) if m.any() else np.full(5, np.nan))
    return np.vstack(out)


def expected_rank(acceptability):
    r = np.arange(1, acceptability.shape[0] + 1)
    return r @ acceptability


def rank_entropy(acceptability):
    b = np.clip(acceptability, 1e-12, 1.0)
    return -(b * np.log(b)).sum(axis=0) / np.log(acceptability.shape[0])
