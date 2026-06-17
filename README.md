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

## Local Run

Run from the repository root:

```bash
python VNStock/module5_vnstock_price.py --output-dir VNStock/data/processed
python module6_session_bucket/export_cafef_from_mongo.py --output data/raw/articles.csv
python module6_session_bucket/session_mapper.py --articles-input data/raw/articles.csv --trading-calendar VNStock/data/processed/daily_sector_price_wide.csv --output data/processed/articles_session_mapped.csv
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

## Automation

The GitHub Actions workflow is:

`.github/workflows/daily_pipeline.yml`

It runs manually through `workflow_dispatch` and daily at 17:30 Vietnam time
(`10:30 UTC`).
