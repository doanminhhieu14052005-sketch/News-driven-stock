# Module 12 — Price Correlation Network (RQ2, part 1)

Builds an undirected **price correlation network** across the 10 HOSE sectors
from their daily return series. This is the first piece of RQ2 (sector
co-movement / spillover structure based on prices rather than sentiment).

## Purpose

- Measure pairwise Pearson correlation between the 10 sector return series.
- Turn that correlation structure into an undirected network: a sector pair is
  connected if their absolute correlation exceeds a threshold.
- Report node-level network metrics (degree, weighted degree, average
  absolute correlation) for each sector.

## Input

`data/processed/merged_sentiment_return_wide.csv` (encoding `utf-8-sig`).

Required columns:
- `trade_date`
- `ret_<sector>` for each of the 10 standard sectors:
  Banking, RealEstate, SteelMaterials, Technology, ConsumerStaples,
  ConsumerDiscretionary, Energy, IndustrialLogistics, Healthcare, Utilities.

`sent_*` columns are ignored.

## Processing

1. Read the 10 `ret_<sector>` columns, coerce to numeric, and drop rows with
   any missing return (listwise; use `--pairwise` to keep pairwise correlation
   instead).
2. Compute the 10x10 Pearson correlation matrix.
3. Build an **undirected** binary adjacency: edge `(i, j) = 1` if
   `|corr_ij| > threshold`. Diagonal is 0.
   - Default threshold = 75th percentile of the absolute off-diagonal
     correlations ("top quartile" of pairs), matching the research design.
   - Use `--threshold <float>` to fix it (e.g. `0.3`).
4. Compute node metrics: `degree`, `weighted_degree` (sum of `|corr|` to
   connected neighbours), and `avg_abs_corr` (mean `|corr|` to all other
   sectors).

## Output (in `data/processed/`, encoding `utf-8-sig`)

- `price_correlation_matrix.csv` — 10x10 correlation values, first column
  `source_sector` (same layout as Module 10 matrices).
- `price_network_adjacency.csv` — 10x10 binary 0/1 adjacency, first column
  `source_sector`.
- `price_network_edges.csv` — edge list `source, target, corr, abs_corr` for
  edges with adjacency = 1; each undirected pair appears once.
- `price_network_node_metrics.csv` — `sector, degree, weighted_degree,
  avg_abs_corr`.

## How to run

From the project root (`E:\News-driven-stock`):

```bash
C:/Users/Admin/anaconda3/envs/tf-gpu/python.exe module12_price_network/build_price_network.py
```

Optional arguments:

```bash
# Fixed correlation threshold instead of the p75 default
... build_price_network.py --threshold 0.3

# Pairwise correlation (do not drop rows listwise)
... build_price_network.py --pairwise

# Override any input/output path
... build_price_network.py --input <path> --edge-list-output <path>
```

The script logs progress, validates the outputs (10x10 matrices, binary &
symmetric adjacency, zero diagonal), prints a summary (rows used, threshold,
edge count, density, top correlated pairs, node metrics) and exits 0 on
success.
