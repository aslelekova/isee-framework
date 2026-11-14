# ISEE: Integrated Sustainable Economic Efficiency

ISEE is a Python research repository for comparing critical mineral systems. It scores each country–mineral pair on financial value, managerial and institutional flexibility, strategic value, environmental burden, and exposure to long-term liabilities.

The included dataset covers 30 systems in 17 countries across copper, lithium, nickel, and cobalt.

## What the code does

- **Ranking sensitivity.** Tests how changes in weights, aggregation rules, and input data affect the rankings of four case studies. [Model comparison](results/table3_specifications.csv) · [Rank distribution plot](figures/fig5_acceptability.png) · [Rank frequencies](results/smaa_acceptability.csv)
- **Clustering.** Uses k-means to group all 30 systems by their five dimension scores and checks how stable those groups are when resampling individual systems or whole countries. [Cluster profiles](results/table7_cluster_profiles.csv) · [Stability metrics](results/cluster_selection_metrics.csv)

## Run the analysis

Use Python 3.11, the version used in [CI](.github/workflows/ci.yml).

```bash
git clone https://github.com/aslelekova/isee-framework.git
cd isee-framework
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_analysis.py
python -m unittest discover -s tests -v
```

The [analysis script](run_analysis.py) uses the code in [`isee/`](isee/) and regenerates the CSV tables in [`results/`](results/) and PNG figures in [`figures/`](figures/).

## Data

The codebook defines the indicators. The provenance file lists sources and reference years. [Dataset](data/mineral_systems.csv) · [Codebook](CODEBOOK.md) · [Data provenance](DATA_PROVENANCE.csv)

## Paper and citation

Companion code and data for *Multidimensional Assessment of Critical Mineral Systems: Financial, Institutional, Strategic and Environmental Factors*, accepted for publication in *Russian Mining Industry*.

See [`CITATION.cff`](CITATION.cff) for the software citation and [`LICENSE`](LICENSE) for the MIT license.
