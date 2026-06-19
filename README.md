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

## Local Run

Run from the repository root:

```bash
python VNStock/module5_vnstock_price.py --output-dir VNStock/data/processed
python module6_session_bucket/export_cafef_from_mongo.py --output data/raw/articles.csv
python module6_session_bucket/session_mapper.py --articles-input data/raw/articles.csv --trading-calendar VNStock/data/processed/daily_sector_price_wide.csv --output data/processed/articles_session_mapped.csv
python module7_sector_sentiment/cafef_sector_sentiment_mapper.py --input data/processed/articles_session_mapped.csv --output data/processed/cafef_articles_labeled.csv --long-output data/processed/cafef_article_sector_long.csv
python module8_daily_sentiment/daily_sector_sentiment_aggregator.py --input data/processed/cafef_article_sector_long.csv --output-long data/processed/daily_sector_sentiment_long.csv --output-wide data/processed/daily_sector_sentiment_wide.csv
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

## Automation

The GitHub Actions workflow is:

`.github/workflows/daily_pipeline.yml`

It runs manually through `workflow_dispatch` and daily at 17:30 Vietnam time
(`10:30 UTC`).
