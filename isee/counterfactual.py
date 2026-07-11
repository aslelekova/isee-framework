import itertools

import numpy as np

from . import smaa
from .data import INDICATOR_UPPER


def rank_probability(X_raw, system_idx, target_rank, n=4_000, seed=42,
                     upper=None):
    res = smaa.sample(X_raw, n=n, noise=0.0, seed=seed, upper=upper)
    return (res["ranks"][:, system_idx] <= target_rank).mean()


def search(X_raw, system_idx, mutable, directions, target_rank, probability,
           max_change=0.5, step=0.05, n=4_000, seed=42):
    grid = np.arange(0.0, max_change + 1e-9, step)
    combos = sorted(itertools.product(grid, repeat=len(mutable)),
                    key=lambda c: sum(c))
    base_p = rank_probability(X_raw, system_idx, target_rank, n=n, seed=seed,
                              upper=INDICATOR_UPPER)
    for combo in combos:
        Xp = X_raw.copy()
        for delta, col, direction in zip(combo, mutable, directions):
            Xp[system_idx, col] = Xp[system_idx, col] * (1.0 + direction * delta)
        Xp = np.clip(Xp, 0.0, INDICATOR_UPPER)
        p = rank_probability(Xp, system_idx, target_rank, n=n, seed=seed,
                             upper=INDICATOR_UPPER)
        if p >= probability:
            return {"changes": dict(zip(mutable, combo)), "probability": p,
                    "baseline_probability": base_p, "total_change": sum(combo)}
    return {"changes": None, "probability": None,
            "baseline_probability": base_p, "total_change": None}
