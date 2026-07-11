import csv
import os
from dataclasses import dataclass, field

import numpy as np

DIMENSIONS = ["FV", "MF", "SV", "EX", "LL"]

DIM_SIGN = {"FV": +1, "MF": +1, "SV": +1, "EX": -1, "LL": -1}

INDICATORS = {
    "FV1: Value of mine production (USD bn)":       ("prod_value_usd_bn", "FV", +1),
    "FV2: Mineral rents (% of GDP)":                ("mineral_rents_pct_gdp", "FV", +1),
    "MF1: Regulatory quality (percentile)":         ("regulatory_quality_pct", "MF", +1),
    "MF2: Government effectiveness (percentile)":   ("gov_effectiveness_pct", "MF", +1),
    "SV1: Share of global mine output (%)":         ("mine_share_pct", "SV", +1),
    "SV2: Share of global refining (%)":            ("refining_share_pct", "SV", +1),
    "EX1: Grid carbon intensity (gCO2/kWh)":        ("grid_co2_g_kwh", "EX", +1),
    "EX2: Water stress (% withdrawal)":             ("water_stress_pct", "EX", +1),
    "LL1: Rule-of-law gap (100 - percentile)":      ("rule_of_law_pct", "LL", -1),
    "LL2: Environmental performance gap (100-EPI)": ("epi_score", "LL", -1),
}

DEFAULT_CSV = os.path.join(os.path.dirname(__file__), "..", "data",
                           "mineral_systems.csv")


@dataclass
class MineralSystems:
    ids: list
    labels: list
    minerals: list
    flagship: np.ndarray
    X: np.ndarray
    indicator_names: list = field(default_factory=lambda: list(INDICATORS))

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
        vals = []
        for _, (col, _, direction) in INDICATORS.items():
            v = float(r[col])
            vals.append(100.0 - v if direction < 0 else v)
        X.append(vals)
    return MineralSystems(ids=ids, labels=labels, minerals=minerals,
                          flagship=np.array(flag), X=np.array(X, float))


def indicator_dimensions():
    return [dim for _, (_, dim, _) in INDICATORS.items()]
