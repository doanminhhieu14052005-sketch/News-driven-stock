# News-driven Sector Sentiment Spillover Network

This repository contains the working pipeline for **News-driven Sector
Sentiment Spillover Network on HOSE**. The project combines Vietnamese
financial news from CafeF with daily sector price returns to support later
Granger causality, lead-lag, and network analysis.

## Project Scope

The current scope uses CafeF as the news source. VNStock is only Module 5
inside the larger project, not the whole project.

## Module 5 - VNStock Price Returns

Location:

`VNStock/module5_vnstock_price.py`

Module 5 fetches daily OHLCV data through `vnstock`, normalizes prices for
sector proxy tickers, and creates equal-weighted sector returns for 10 HOSE
sector groups.

Main outputs:

- `VNStock/data/processed/daily_sector_price_wide.csv`
- `VNStock/data/processed/daily_sector_price_long.csv`

## Module 6 - Session Bucket Mapper

Location:

`module6_session_bucket/session_mapper.py`

Module 6 maps CafeF article `published_at` timestamps to:

- `published_at_vn`
- `session_bucket`
- `trade_date`
- `mapped_reason`
- `is_mapped`

It uses the Module 5 trading calendar from:

`VNStock/data/processed/daily_sector_price_wide.csv`

CafeF articles can be exported from MongoDB with:

`module6_session_bucket/export_cafef_from_mongo.py`

## Module 7 - CafeF Sector & Sentiment Mapper

Location:

`module7_sector_sentiment/cafef_sector_sentiment_mapper.py`

Module 7 normalizes raw CafeF `impact`, `sector`, and ticker information into
`sentiment_score` and standard `sector_label` values for the 10 HOSE sector
groups used by Module 5.

If an article has valid ticker information, ticker-to-sector mapping is used
first. If ticker information is missing, Module 7 tries sector keywords. If
there is still not enough information, the article remains `Unmapped` for audit
and is not used in sector-level aggregation. Broad market or macroeconomic news
is marked `MarketMacro` at article level and also excluded from 10-sector
aggregation.

## Module 8 - Daily Sector Sentiment Aggregation

Location:

`module8_daily_sentiment/daily_sector_sentiment_aggregator.py`

Module 8 aggregates article-sector sentiment from Module 7 into daily
sector-level sentiment time series. It creates long and wide outputs for later
Granger causality, lead-lag, and network comparison steps.

Missing news is kept as `NaN`, not filled with zero. Missing sentiment is not
treated as neutral.

## Module 9 - Merge Daily Sentiment with Sector Returns

Location:

`module9_merge_sentiment_return/merge_sentiment_return.py`

Module 9 merges daily sector sentiment with sector returns using the common
trading-date window. It creates merged wide/long datasets and a coverage report
for downstream Granger causality and network analysis.

Missing sentiment is kept as `NaN`, not filled with zero. Missing returns are
also preserved.

## Module 10 - Granger Causality Analysis

Location:

`module10_granger/granger_causality.py`

Module 10 runs Augmented Dickey-Fuller stationarity diagnostics and pairwise
Granger causality tests from sector sentiment to sector returns:

`sent_<source_sector> -> ret_<target_sector>`

For each source-target pair, the selected lag is the lag with the minimum
p-value among valid lags. P-values are corrected with Benjamini-Hochberg FDR
across successful sector pairs. Missing sentiment and return values are not
filled; each pair is tested after pairwise `dropna()`.

Granger causality here means predictive causality in a time-series sense. It
does not prove structural or economic causality.

## Local Run

Run from the repository root:

```bash
python VNStock/module5_vnstock_price.py --output-dir VNStock/data/processed
python module6_session_bucket/export_cafef_from_mongo.py --output data/raw/articles.csv
python module6_session_bucket/session_mapper.py --articles-input data/raw/articles.csv --trading-calendar VNStock/data/processed/daily_sector_price_wide.csv --output data/processed/articles_session_mapped.csv
python module7_sector_sentiment/cafef_sector_sentiment_mapper.py --input data/processed/articles_session_mapped.csv --output data/processed/cafef_articles_labeled.csv --long-output data/processed/cafef_article_sector_long.csv
python module8_daily_sentiment/daily_sector_sentiment_aggregator.py --input data/processed/cafef_article_sector_long.csv --output-long data/processed/daily_sector_sentiment_long.csv --output-wide data/processed/daily_sector_sentiment_wide.csv
python module9_merge_sentiment_return/merge_sentiment_return.py --price-wide VNStock/data/processed/daily_sector_price_wide.csv --sentiment-wide data/processed/daily_sector_sentiment_wide.csv --sentiment-long data/processed/daily_sector_sentiment_long.csv --output-wide data/processed/merged_sentiment_return_wide.csv --output-long data/processed/merged_sentiment_return_long.csv --coverage-report data/processed/merge_coverage_report.csv
python module10_granger/granger_causality.py --input data/processed/merged_sentiment_return_wide.csv --stationarity-output data/processed/stationarity_tests.csv --all-lags-output data/processed/granger_all_lags.csv --results-output data/processed/granger_results.csv --pvalue-matrix-output data/processed/granger_pvalue_matrix.csv --fdr-matrix-output data/processed/granger_fdr_matrix.csv --adjacency-output data/processed/granger_adjacency_matrix.csv --edge-list-output data/processed/granger_edge_list.csv --max-lag 5 --min-observations 60 --alpha 0.05
```

## GitHub Secrets

Add these secrets in GitHub before running the daily workflow:

- `MONGO_URI`
- `MONGO_DB_NAME`
- `MONGO_COLLECTION`

Do not commit `.env`, MongoDB usernames, passwords, API keys, or other
secrets. `data/raw/articles.csv` is generated from MongoDB and is ignored by
Git.

## Pipeline Outputs

- `VNStock/data/processed/daily_sector_price_wide.csv`
- `VNStock/data/processed/daily_sector_price_long.csv`
- `data/processed/articles_session_mapped.csv`
- `data/processed/cafef_articles_labeled.csv`
- `data/processed/cafef_article_sector_long.csv`
- `data/processed/daily_sector_sentiment_long.csv`
- `data/processed/daily_sector_sentiment_wide.csv`
- `data/processed/merged_sentiment_return_wide.csv`
- `data/processed/merged_sentiment_return_long.csv`
- `data/processed/merge_coverage_report.csv`
- `data/processed/stationarity_tests.csv`
- `data/processed/granger_all_lags.csv`
- `data/processed/granger_results.csv`
- `data/processed/granger_pvalue_matrix.csv`
- `data/processed/granger_fdr_matrix.csv`
- `data/processed/granger_adjacency_matrix.csv`
- `data/processed/granger_edge_list.csv`

## Automation

The GitHub Actions workflow is:

`.github/workflows/daily_pipeline.yml`

It runs manually through `workflow_dispatch` and daily at 17:30 Vietnam time
(`10:30 UTC`).
