# ISEE: Integrated Sustainable Economic Efficiency framework

Reference implementation and data for the paper *"Beyond Financial Metrics:
Reframing Economic Efficiency in Critical Mineral Supply Chains under the
Global Energy Transition"*. The package computes the ISEE composite indicator,
performs uncertainty and sensitivity analysis (alternative weighting schemes,
weight and input-perturbation Monte Carlo, leave-one-indicator-out),
analyses compensability through generalised-mean aggregation, computes SMAA
rank acceptability, pairwise outranking probabilities, and central weights,
runs counterfactual policy analysis, and derives a cluster typology of 30
country–mineral systems.

## Structure

```
isee_framework/
├── data/
│   └── mineral_systems.csv     # 30 country–mineral systems (see CODEBOOK.md)
├── isee/
│   ├── data.py                 # dataset loading, indicator construction
│   ├── normalise.py            # min-max / z-score / percentile normalisation
│   ├── weights.py              # equal, entropy, PCA, CRITIC weighting
│   ├── aggregate.py            # additive and geometric aggregation, ranks
│   ├── robustness.py           # Monte Carlo, LOIO, correlation analysis
│   ├── smaa.py                 # generalised-mean aggregation, SMAA rank acceptability
│   ├── counterfactual.py       # counterfactual policy analysis
│   ├── typology.py             # k-means, Ward, silhouette, bootstrap stability
│   └── figures.py              # publication figures
├── run_analysis.py             # reproduces all tables and figures of Section 5
├── tests/
│   └── test_isee.py            # 30 unit and regression tests
├── CODEBOOK.md                 # indicator definitions and formulas
├── DATA_PROVENANCE.csv         # source, URL, and vintage of every column
└── requirements.txt            # pinned dependency versions
```

## Usage

Python ≥ 3.9.

```bash
pip install -r requirements.txt
python3 run_analysis.py          # writes ./results/*.csv and ./figures/*.png
python3 -m unittest discover tests -v
```

## Data

Each column of `data/mineral_systems.csv` is documented in `CODEBOOK.md`
(definition, unit, construction formula) and `DATA_PROVENANCE.csv` (primary
source, URL, vintage). Indicator values are compiled from the public sources
listed there; the pipeline is fully data-driven, so revised values only
require re-running `run_analysis.py`.

## Reproducibility

All stochastic procedures (Monte Carlo simulations, k-means restarts,
bootstrap resampling) use fixed random seeds; repeated runs produce identical
results. The test suite includes regression tests that pin the published
headline numbers (baseline ISEE scores, entropy weights, Monte Carlo rank
frequencies, selected number of clusters). Continuous integration runs the
tests and the full analysis on every push.

## License and citation

MIT License (see `LICENSE`). Citation metadata is provided in `CITATION.cff`.
