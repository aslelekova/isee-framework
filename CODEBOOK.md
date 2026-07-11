# Codebook

Unit of observation: a country–mineral system (a country's position in the
global supply chain of one focal mineral). The dataset covers 30 systems:
the principal producers of copper (11), lithium (6), nickel (6), and
cobalt (7). `flagship = 1` marks the four case-study systems analysed in
Section 4 of the paper.

## Raw columns (`data/mineral_systems.csv`)

| Column | Definition | Unit |
|---|---|---|
| `system_id` | ISO3 code + mineral symbol | — |
| `prod_value_usd_bn` | 2023 mine output x average 2023 world price | USD bn |
| `gdp_usd_bn` | GDP, current USD, 2023 | USD bn |
| `regulatory_quality_pct` | WGI Regulatory Quality percentile rank, 2022 | 0–100 |
| `gov_effectiveness_pct` | WGI Government Effectiveness percentile rank, 2022 | 0–100 |
| `mine_share_pct` | share of global mine production of the focal mineral, 2023 | % |
| `refining_share_pct` | share of global refining/processing of the focal mineral, 2023 | % |
| `grid_co2_g_kwh` | carbon intensity of electricity generation, 2023 | gCO2/kWh |
| `water_stress_pct` | freshwater withdrawal / available resources (SDG 6.4.2) | % |
| `rule_of_law_pct` | WGI Rule of Law percentile rank, 2022 | 0–100 |
| `epi_score` | Yale Environmental Performance Index, 2022 | 0–100 |

## Derived indicators (constructed in `isee/data.py`)

| Indicator | Dimension | Formula |
|---|---|---|
| FV1 | Financial Value | `prod_value_usd_bn` |
| FV2 | Financial Value | `100 * prod_value_usd_bn / gdp_usd_bn` |
| MF1 | Managerial Flexibility | `regulatory_quality_pct` |
| MF2 | Managerial Flexibility | `gov_effectiveness_pct` |
| SV1 | Strategic Value | `refining_share_pct` |
| SV2 | Strategic Value | `refining_share_pct / mine_share_pct` (downstream capture ratio) |
| EX1 | Environmental operating context | `grid_co2_g_kwh` |
| EX2 | Environmental operating context | `water_stress_pct` |
| LL1 | Liability exposure / enforcement | `100 - rule_of_law_pct` |
| LL2 | Liability exposure / enforcement | `100 - epi_score` |

FV2 is constructed from the focal mineral's production value rather than from
World Bank mineral rents, because the World Bank aggregate excludes lithium
and cobalt and is not mineral-specific.

EX1/EX2 are national-context proxies: they describe the environmental
operating environment of the jurisdiction, not the site-level footprint of
the focal mining operations.

LL1/LL2 are institutional proxies for long-term liability exposure and
enforcement capacity, not estimates of closure or rehabilitation obligations.

## Prices used for FV1 (average 2023, USD/t)

| Mineral | Price | Basis |
|---|---|---|
| Copper | 8,490 | LME cash, annual average |
| Nickel | 21,500 | LME cash, annual average |
| Cobalt | 33,000 | LME/Fastmarkets, annual average |
| Lithium | 35,000 per t LCE | lithium carbonate, annual average; Li content x 5.32 = LCE |
