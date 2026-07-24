import csv
import os
from dataclasses import dataclass, field

import numpy as np

DIMENSIONS = ["FV", "MF", "SV", "EX", "LL"]

DIM_SIGN = {"FV": +1, "MF": +1, "SV": +1, "EX": -1, "LL": -1}

INDICATOR_NAMES = [
    "FV1: Value of mine production (USD bn)",
    "FV2: Focal-mineral production value (% of GDP)",
    "MF1: Regulatory quality (percentile)",
    "MF2: Government effectiveness (percentile)",
    "SV1: Share of global refining (%)",
    "SV2: Downstream capture ratio (refining/mining share)",
    "EX1: Grid carbon intensity (gCO2/kWh)",
    "EX2: Water stress (% withdrawal)",
    "LL1: Rule-of-law gap (100 - percentile)",
    "LL2: Environmental performance gap (100 - EPI)",
]

INDICATOR_DIMS = ["FV", "FV", "MF", "MF", "SV", "SV", "EX", "EX", "LL", "LL"]

PRIMITIVE_NAMES = [
    "prod_value_usd_bn", "gdp_usd_bn", "regulatory_quality_pct",
    "gov_effectiveness_pct", "mine_share_pct", "refining_share_pct",
    "grid_co2_g_kwh", "water_stress_pct", "rule_of_law_pct", "epi_score",
]

PRIMITIVE_UPPER = np.array([np.inf, np.inf, 100.0, 100.0, 100.0, 100.0,
                            np.inf, 100.0, 100.0, 100.0])

PRIMITIVE_COUNTRY_LEVEL = np.array([False, True, True, True, False, False,
                                    True, True, True, True])

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                           "mineral_systems.csv")


def build_indicators(P):
    pv, gdp, rq, ge, mine, ref, grid, water, rl, epi = np.asarray(P, float).T
    return np.column_stack([
        pv,
        100.0 * pv / gdp,
        rq,
        ge,
        ref,
        ref / mine,
        grid,
        water,
        100.0 - rl,
        100.0 - epi,
    ])


@dataclass
class MineralSystems:
    ids: list
    labels: list
    minerals: list
    countries: list
    flagship: np.ndarray
    P: np.ndarray
    X: np.ndarray
    indicator_names: list = field(default_factory=lambda: list(INDICATOR_NAMES))

    def __len__(self):
        return len(self.ids)

    def subset(self, mask):
        mask = np.asarray(mask, bool)
        return MineralSystems(
            ids=[i for i, m in zip(self.ids, mask) if m],
            labels=[l for l, m in zip(self.labels, mask) if m],
            minerals=[c for c, m in zip(self.minerals, mask) if m],
            countries=[c for c, m in zip(self.countries, mask) if m],
            flagship=self.flagship[mask],
            P=self.P[mask],
            X=self.X[mask],
        )


def load(path=DEFAULT_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids, labels, minerals, countries, flag, P = [], [], [], [], [], []
    for r in rows:
        ids.append(r["system_id"])
        labels.append(f"{r['country']} ({r['mineral']})")
        minerals.append(r["mineral"])
        countries.append(r["iso3"])
        flag.append(r["flagship"] == "1")
        P.append([float(r[c]) for c in PRIMITIVE_NAMES])
    P = np.array(P, float)
    return MineralSystems(ids=ids, labels=labels, minerals=minerals,
                          countries=countries, flagship=np.array(flag),
                          P=P, X=build_indicators(P))


def indicator_dimensions():
    return list(INDICATOR_DIMS)


def perturb_primitives(P, countries, rng, noise):
    F = rng.uniform(1.0 - noise, 1.0 + noise, size=P.shape)
    first = {}
    for j, c in enumerate(countries):
        if c in first:
            F[j, PRIMITIVE_COUNTRY_LEVEL] = F[first[c], PRIMITIVE_COUNTRY_LEVEL]
        else:
            first[c] = j
    return np.clip(P * F, 0.0, PRIMITIVE_UPPER)
