import itertools

import numpy as np

from . import smaa
from .data import MineralSystems, PRIMITIVE_UPPER, build_indicators

LEVERS = {
    "grid_co2": 6,
    "water_stress": 7,
    "rule_of_law_gap": 8,
    "epi_gap": 9,
}

GAP_LEVERS = {"rule_of_law_gap", "epi_gap"}


def apply_levers(ms, system_idx, changes):
    P = ms.P.copy()
    for name, delta in changes.items():
        col = LEVERS[name]
        if name in GAP_LEVERS:
            P[system_idx, col] = 100.0 - (100.0 - P[system_idx, col]) * (1.0 - delta)
        else:
            P[system_idx, col] = P[system_idx, col] * (1.0 - delta)
    P = np.clip(P, 0.0, PRIMITIVE_UPPER)
    return MineralSystems(ids=ms.ids, labels=ms.labels, minerals=ms.minerals,
                          countries=ms.countries, flagship=ms.flagship,
                          P=P, X=build_indicators(P))


def rank_probability(ms, system_idx, target_rank, n=4_000, seed=42):
    res = smaa.sample(ms, n=n, noise=0.0, seed=seed)
    return (res["ranks"][:, system_idx] <= target_rank).mean()


def scenario_search(ms, system_idx, levers, target_rank, probability,
                    costs=None, max_change=0.5, step=0.05, n=4_000, seed=42):
    costs = costs or {name: 1.0 for name in levers}
    grid = np.arange(0.0, max_change + 1e-9, step)
    combos = sorted(itertools.product(grid, repeat=len(levers)),
                    key=lambda c: sum(cost * d for cost, d in
                                      zip((costs[l] for l in levers), c)))
    base_p = rank_probability(ms, system_idx, target_rank, n=n, seed=seed)
    for combo in combos:
        changes = dict(zip(levers, combo))
        p = rank_probability(apply_levers(ms, system_idx, changes),
                             system_idx, target_rank, n=n, seed=seed)
        if p >= probability:
            return {"changes": changes, "probability": float(p),
                    "baseline_probability": float(base_p),
                    "cost": float(sum(costs[l] * d
                                      for l, d in changes.items()))}
    return {"changes": None, "probability": None,
            "baseline_probability": float(base_p), "cost": None}
