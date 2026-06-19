# Module 11 - Granger Spillover Network Construction and Visualization

## Purpose

Module 11 builds directed Granger sentiment spillover networks from Module 10
outputs. It does not re-run stationarity tests or Granger causality tests.

The edge direction is:

`source_sector sentiment -> target_sector return`

This is a predictive spillover network. It is not proof of an absolute causal
mechanism.

## Inputs

Default inputs:

- `data/processed/granger_results.csv`
- `data/processed/granger_pvalue_matrix.csv`
- `data/processed/granger_fdr_matrix.csv`
- `data/processed/granger_adjacency_matrix.csv`

All CSV files are read with `utf-8-sig`.

## Outputs

CSV outputs:

- `data/processed/network_edges_raw.csv`
- `data/processed/network_edges_fdr.csv`
- `data/processed/network_node_metrics_raw.csv`
- `data/processed/network_node_metrics_fdr.csv`
- `data/processed/network_summary.csv`

Figure outputs:

- `reports/figures/granger_network_raw.png`
- `reports/figures/granger_network_fdr.png`
- `reports/figures/granger_pvalue_heatmap.png`
- `reports/figures/granger_fdr_heatmap.png`
- `reports/figures/granger_raw_adjacency_heatmap.png`
- `reports/figures/granger_fdr_adjacency_heatmap.png`

All output CSV files are written with `utf-8-sig`.

## Raw vs FDR Network

The raw network uses rows where `significant_raw_05 == True`. It is useful for
exploratory analysis.

The FDR network uses rows where `significant_fdr_05 == True`. It is more
conservative and may be empty with the current data. If there are no FDR edges,
Module 11 still writes empty edge CSVs, full 10-sector node metrics, and a
network figure with all nodes plus a no-edge message.

## Self-Loop Handling

Self-loops are kept in the edge CSV files because own-sector predictability is
informative.

By default, network metrics and network plots use cross-sector edges only.
Use `--include-self-loops-in-plot` to include self-loops in network figures.

## Node Metrics

Each node metrics file contains one row per standard sector:

- `in_degree`
- `out_degree`
- `total_degree`
- `weighted_in_degree`
- `weighted_out_degree`
- `weighted_total_degree`
- `self_loop_count`
- `pagerank`
- `betweenness_centrality`
- `is_net_sender`
- `is_net_receiver`

Degrees and weighted degrees are based on cross-sector edges. `self_loop_count`
is reported separately.

## Local Run

Run from the repository root:

```bash
python module11_network/build_granger_network.py \
  --granger-results data/processed/granger_results.csv \
  --pvalue-matrix data/processed/granger_pvalue_matrix.csv \
  --fdr-matrix data/processed/granger_fdr_matrix.csv \
  --adjacency-matrix data/processed/granger_adjacency_matrix.csv \
  --output-dir data/processed \
  --figures-dir reports/figures \
  --alpha 0.05
```

Optional flags:

- `--include-self-loops-in-plot`
- `--top-n-labels 10`
- `--min-edge-weight 0.0`

## Reading the Figures

The network figures show sector nodes and directed edges from source sentiment
to target return. Wider edges have larger `-log10(p-value)` weights.

The p-value heatmaps use source sectors as rows and target sectors as columns.
Lower p-values indicate stronger predictive evidence under the tested Granger
model.
