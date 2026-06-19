# Module 10 - Stationarity and Granger Causality Analysis

## Purpose

Module 10 runs stationarity diagnostics and Granger causality tests on the
merged daily sentiment-return dataset from Module 9. It prepares statistical
tables and matrices for Module 11 network construction.

Module 10 does not draw network visualizations.

## Input

Default input:

`data/processed/merged_sentiment_return_wide.csv`

Expected columns:

- `trade_date`
- `ret_<sector>` for the 10 standard sectors
- `sent_<sector>` for the 10 standard sectors

## Outputs

- `data/processed/stationarity_tests.csv`
- `data/processed/granger_all_lags.csv`
- `data/processed/granger_results.csv`
- `data/processed/granger_pvalue_matrix.csv`
- `data/processed/granger_fdr_matrix.csv`
- `data/processed/granger_adjacency_matrix.csv`
- `data/processed/granger_edge_list.csv`

All CSV files are written with `utf-8-sig`.

## Stationarity Test

The module runs Augmented Dickey-Fuller tests with `statsmodels.adfuller` for
all return and sentiment variables. It records observation count, missing ratio,
ADF statistic, p-value, used lag, and a 5 percent stationarity flag.

The module does not automatically difference or transform the data.

## Granger Direction

The tested direction is:

`sent_<source_sector> -> ret_<target_sector>`

In `statsmodels.grangercausalitytests`, the target return column is placed first
and the source sentiment column second.

## Lag Selection

Tests are run for lags from 1 to `--max-lag`, subject to a conservative
observation-based effective maximum lag. The selected lag for each pair is the
lag with the smallest valid p-value.

## FDR Correction

Benjamini-Hochberg FDR correction is applied to successful pair-level p-values.
The adjacency matrix uses FDR-significant pairs at `--alpha`.

`edge_weight = -log10(p_value_fdr_bh)`.

## Missing Values

Missing sentiment and return values are not filled with zero and are not
forward-filled. Each Granger pair uses pairwise `dropna` on only the target
return and source sentiment columns.

## Local Run

```bash
python module10_granger/granger_causality.py --input data/processed/merged_sentiment_return_wide.csv --stationarity-output data/processed/stationarity_tests.csv --all-lags-output data/processed/granger_all_lags.csv --results-output data/processed/granger_results.csv --pvalue-matrix-output data/processed/granger_pvalue_matrix.csv --fdr-matrix-output data/processed/granger_fdr_matrix.csv --adjacency-output data/processed/granger_adjacency_matrix.csv --edge-list-output data/processed/granger_edge_list.csv --max-lag 5 --min-observations 60 --alpha 0.05
```

Optional flags:

- `--exclude-self`
- `--start-date YYYY-MM-DD`
- `--end-date YYYY-MM-DD`

## Interpretation

Granger causality here means predictive causality. A significant edge indicates
that past source-sector sentiment improves prediction of target-sector returns
under the tested model. It is not proof of absolute causal mechanism.
