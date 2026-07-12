# Codebook

Unit of observation: a country–mineral system (a country's position in the
global supply chain of one focal mineral). The dataset covers 30 systems:
the principal producers of copper (11), lithium (6), nickel (6), and
cobalt (7). `flagship = 1` marks the four case-study systems analysed in
Section 4 of the paper. Reference period: the most recent year available per
source at the time of compilation (July 2026); per-column vintages, sources,
and access dates are listed in `DATA_PROVENANCE.csv`.

## Raw columns (`data/mineral_systems.csv`)

| Column | Definition | Unit |
|---|---|---|
| `system_id` | ISO3 code + mineral symbol | — |
| `prod_value_usd_bn` | 2025 mine output (USGS MCS 2026) x average 2025 world price | USD bn |
| `gdp_usd_bn` | GDP, current USD, 2025 (Cuba: 2020, latest available) | USD bn |
| `regulatory_quality_score` | WGI Regulatory Quality anchored score, 2025 revision, obs. 2024 | 0–100 |
| `gov_effectiveness_score` | WGI Government Effectiveness anchored score, 2025 revision, obs. 2024 | 0–100 |
| `mine_share_pct` | share of global mine production of the focal mineral, 2025 | % |
| `refining_share_pct` | share of global refining/processing activity of the focal mineral, 2024–2025; the underlying basis differs by mineral (reported refinery production for copper, refining capacity for nickel, approximate processing shares for lithium and cobalt) | % |
| `grid_co2_g_kwh` | carbon intensity of electricity generation (Ember), 2024–2025 | gCO2/kWh |
| `water_stress_pct` | freshwater withdrawal / available resources (SDG 6.4.2), 2022 | % |
| `rule_of_law_score` | WGI Rule of Law anchored score, 2025 revision, obs. 2024 | 0–100 |
| `epi_score` | Yale Environmental Performance Index, 2026 edition (blank = not covered) | 0–100 |

The 2025 WGI revision reports anchored 0–100 governance scores instead of
the percentile ranks used in earlier releases.

## Derived indicators (constructed in `isee/data.py`)

| Indicator | Dimension | Formula |
|---|---|---|
| FV1 | Financial Value | `prod_value_usd_bn` |
| FV2 | Financial Value | `100 * prod_value_usd_bn / gdp_usd_bn` |
| MF1 | Managerial Flexibility | `regulatory_quality_score` |
| MF2 | Managerial Flexibility | `gov_effectiveness_score` |
| SV1 | Strategic Value | `refining_share_pct` |
| SV2 | Strategic Value | `refining_share_pct / mine_share_pct` (downstream capture ratio) |
| EX1 | Environmental operating context | `grid_co2_g_kwh` |
| EX2 | Environmental operating context | `water_stress_pct` |
| LL1 | Liability exposure / enforcement | `100 - rule_of_law_score` |
| LL2 | Liability exposure / enforcement | `100 - epi_score` |

FV2 is constructed from the focal mineral's production value rather than from
World Bank mineral rents, because the World Bank aggregate excludes lithium
and cobalt and is not mineral-specific.

EX1/EX2 are national-context proxies: they describe the environmental
operating environment of the jurisdiction, not the site-level footprint of
the focal mining operations.

LL1/LL2 are institutional proxies for long-term liability exposure and
enforcement capacity, not estimates of closure or rehabilitation obligations.

Missing values (Cuba is absent from the 2026 EPI) propagate as NaN; dimension
scores average the available indicators of the dimension (`np.nanmean`), so
Cuba's LL dimension equals its rule-of-law gap alone. No values are imputed.

## Prices used for FV1 (average 2025, per USGS MCS 2026)

| Mineral | Price | Basis |
|---|---|---|
| Copper | 9,700 USD/t | LME grade A cash (440 c/lb), annual average |
| Nickel | 15,000 USD/t | LME cash, annual average |
| Cobalt | 33,000 USD/t | LME cash (15 USD/lb), annual average |
| Lithium | 9,000 USD/t LCE | battery-grade lithium carbonate, annual average; Li content x 5.323 = LCE |
