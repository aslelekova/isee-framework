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

INDICATOR_UPPER = np.array([np.inf, 100.0, 100.0, 100.0, 100.0, np.inf,
                            np.inf, 100.0, 100.0, 100.0])

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                           "mineral_systems.csv")


@dataclass
class MineralSystems:
    ids: list
    labels: list
    minerals: list
    flagship: np.ndarray
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
            flagship=self.flagship[mask],
            X=self.X[mask],
        )


def load(path=DEFAULT_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    ids, labels, minerals, flag, X = [], [], [], [], []
    for r in rows:
        ids.append(r["system_id"])
        labels.append(f"{r['country']} ({r['mineral']})")
        minerals.append(r["mineral"])
        flag.append(r["flagship"] == "1")
        g = lambda c: float(r[c])
        X.append([
            g("prod_value_usd_bn"),
            100.0 * g("prod_value_usd_bn") / g("gdp_usd_bn"),
            g("regulatory_quality_pct"),
            g("gov_effectiveness_pct"),
            g("refining_share_pct"),
            g("refining_share_pct") / g("mine_share_pct"),
            g("grid_co2_g_kwh"),
            g("water_stress_pct"),
            100.0 - g("rule_of_law_pct"),
            100.0 - g("epi_score"),
        ])
    return MineralSystems(ids=ids, labels=labels, minerals=minerals,
                          flagship=np.array(flag), X=np.array(X, float))


def indicator_dimensions():
    return list(INDICATOR_DIMS)
