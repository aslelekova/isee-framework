import numpy as np
from scipy.optimize import brentq

from . import aggregate, normalise
from .data import build_indicators, perturb_primitives

FLOOR = 0.1


def power_mean(S, w, rho, floor=FLOOR):
    B = floor + (1.0 - floor) * aggregate.benefit_oriented(S)
    w = np.asarray(w, float)
    if abs(rho) < 1e-9:
        return np.exp(np.log(B) @ w)
    return (np.power(B, rho) @ w) ** (1.0 / rho)


def compensability_sweep(S, w, rho_grid=None, floor=FLOOR):
    if rho_grid is None:
        rho_grid = np.arange(-2.0, 2.0 + 1e-9, 0.05)
    scores = np.vstack([power_mean(S, w, r, floor) for r in rho_grid])
    ranks = (-scores).argsort(axis=1).argsort(axis=1) + 1
    return np.asarray(rho_grid), scores, ranks


def crossover(S, w, i, j, floor=FLOOR, lo=-2.0, hi=2.0):
    f = lambda rho: (power_mean(S, w, rho, floor)[i]
                     - power_mean(S, w, rho, floor)[j])
    if f(lo) * f(hi) > 0:
        return None
    return brentq(f, lo, hi, xtol=1e-6)


def sample(ms, n=20_000, alpha=1.0, rho_range=(-2.0, 2.0), noise=0.10,
           seed=42):
    rng = np.random.default_rng(seed)
    n_sys = len(ms)
    W = rng.dirichlet(np.full(5, alpha), size=n)
    rhos = rng.uniform(*rho_range, size=n)
    scores = np.empty((n, n_sys))
    for i in range(n):
        if noise > 0:
            Pp = perturb_primitives(ms.P, ms.countries, rng, noise)
            Xp = build_indicators(Pp)
        else:
            Xp = ms.X
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


def central_profile(sample_result, rank=1, q=(10, 90)):
    ranks = sample_result["ranks"]
    W, rhos = sample_result["weights"], sample_result["rhos"]
    out = []
    for j in range(ranks.shape[1]):
        m = ranks[:, j] == rank
        if not m.any():
            out.append(None)
            continue
        out.append({
            "n": int(m.sum()),
            "w_mean": W[m].mean(axis=0),
            "w_lo": np.percentile(W[m], q[0], axis=0),
            "w_hi": np.percentile(W[m], q[1], axis=0),
            "rho_mean": float(rhos[m].mean()),
            "rho_median": float(np.median(rhos[m])),
            "rho_lo": float(np.percentile(rhos[m], q[0])),
            "rho_hi": float(np.percentile(rhos[m], q[1])),
        })
    return out


def expected_rank(acceptability):
    r = np.arange(1, acceptability.shape[0] + 1)
    return r @ acceptability


def rank_entropy(acceptability):
    b = np.clip(acceptability, 1e-12, 1.0)
    return -(b * np.log(b)).sum(axis=0) / np.log(acceptability.shape[0])
