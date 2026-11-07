# ISEE: Integrated Sustainable Economic Efficiency framework

Research code and data accompanying *"Multidimensional Assessment of Critical
Mineral Systems: Financial, Institutional, Strategic and Environmental Factors"*.
The article has been accepted for publication in *Russian Mining Industry*,
issue 3 (May-June 2027).

The package computes the ISEE composite indicator,
performs uncertainty and sensitivity analysis (alternative weighting schemes,
weight and input-perturbation Monte Carlo, leave-one-indicator-out),
analyses compensability through generalised-mean aggregation, computes SMAA
rank acceptability, pairwise outranking probabilities, and central weights,
runs counterfactual policy analysis, and derives a cluster typology of 30
country–mineral systems.

## Structure

```
isee-framework/
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
│   └── test_isee.py            # unit and regression tests
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
results. The test suite includes regression tests that pin the reference
results (baseline ISEE scores, entropy weights, Monte Carlo rank
frequencies, selected number of clusters). Continuous integration runs the
tests and the full analysis on every push.

## License and citation

MIT License (see `LICENSE`). Software citation metadata is provided in
[`CITATION.cff`](CITATION.cff); the article title and publication status are
listed above.
