# ISEE: Integrated Sustainable Economic Efficiency framework

Reference implementation and data for the paper *"Beyond Financial Metrics:
Reframing Economic Efficiency in Critical Mineral Supply Chains under the
Global Energy Transition"*. The package computes the ISEE composite indicator,
performs uncertainty and sensitivity analysis, and derives a machine-learning
typology of 30 country–mineral systems.

## Structure

```
isee_framework/
├── data/
│   └── mineral_systems.csv     # 30 country–mineral systems x 10 indicators
├── isee/
│   ├── data.py                 # dataset loading, indicator definitions
│   ├── normalise.py            # min-max / z-score / percentile normalisation
│   ├── weights.py              # equal, entropy, PCA, CRITIC weighting
│   ├── aggregate.py            # additive and geometric aggregation, ranks
│   ├── robustness.py           # Monte Carlo over weights and over input data
│   ├── typology.py             # k-means, Ward clustering, silhouette, PCA
│   └── figures.py              # publication figures
├── run_analysis.py             # reproduces all tables and figures of Section 5
├── tests/
│   └── test_isee.py            # unit tests (stdlib unittest)
└── requirements.txt
```

## Usage

```bash
pip install -r requirements.txt
python3 run_analysis.py          # writes ./results/*.csv and ./figures/*.png
python3 -m unittest discover tests -v
```

## Data dictionary (`data/mineral_systems.csv`)

| Column | Indicator | Source |
|---|---|---|
| `prod_value_usd_bn` | FV1: value of 2023 mine production (output x average 2023 world price) | USGS Mineral Commodity Summaries 2024; annual average prices |
| `mineral_rents_pct_gdp` | FV2: mineral rents, % of GDP | World Bank WDI (NY.GDP.MINR.RT.ZS), 2021 |
| `regulatory_quality_pct` | MF1: regulatory quality, percentile rank | Worldwide Governance Indicators, 2022 |
| `gov_effectiveness_pct` | MF2: government effectiveness, percentile rank | Worldwide Governance Indicators, 2022 |
| `mine_share_pct` | SV1: share of global mine production, % | USGS Mineral Commodity Summaries 2024 |
| `refining_share_pct` | SV2: share of global refining/processing, % | IEA Global Critical Minerals Outlook 2024 |
| `grid_co2_g_kwh` | EX1: grid carbon intensity, gCO2/kWh | Ember, 2023 |
| `water_stress_pct` | EX2: freshwater withdrawal / available resources (SDG 6.4.2), % | World Bank WDI (ER.H2O.FWST.ZS) |
| `rule_of_law_pct` | LL1 = 100 - value: rule-of-law gap | Worldwide Governance Indicators, 2022 |
| `epi_score` | LL2 = 100 - value: environmental performance gap | Yale Environmental Performance Index, 2022 |

`flagship = 1` marks the four case-study systems analysed in Section 4 of the
paper (Chile–copper, Australia–lithium, Indonesia–nickel, DR Congo–cobalt).

**Note.** Indicator values were compiled from the public sources listed above
and should be re-verified against the original databases before final
submission; the analysis pipeline is fully data-driven, so corrected values
only require re-running `run_analysis.py`.

## Reproducibility

All stochastic procedures (Monte Carlo simulations, k-means restarts) use
fixed random seeds; repeated runs produce identical results.
