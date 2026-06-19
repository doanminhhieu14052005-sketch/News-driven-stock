# Module 9 - Merge Daily Sentiment with Sector Returns

## Purpose

Module 9 merges daily sector sentiment from Module 8 with daily sector returns
from Module 5. It prepares clean merged datasets and diagnostics for later
Granger causality and network analysis.

Module 9 does not run Granger causality.

## Inputs

Default inputs:

- `VNStock/data/processed/daily_sector_price_wide.csv`
- `data/processed/daily_sector_sentiment_wide.csv`
- `data/processed/daily_sector_sentiment_long.csv`

All inputs are read with `utf-8-sig`.

## Outputs

- `data/processed/merged_sentiment_return_wide.csv`
- `data/processed/merged_sentiment_return_long.csv`
- `data/processed/merge_coverage_report.csv`

The wide output has one row per trading day and 20 sector columns:

- `ret_<sector>` for sector returns
- `sent_<sector>` for weighted sector sentiment

The long output has one row per `trade_date` and `sector_label`.

## Merge Logic

1. Parse `trade_date` in price and sentiment inputs.
2. Compute the overlap window between price and sentiment date ranges.
3. Optionally apply `--start-date` and `--end-date` inside that overlap.
4. Use the price trading calendar as the base.
5. Left join sentiment by `trade_date`.
6. Keep missing sentiment and return values as `NaN`.

No missing sentiment or return values are filled with zero. No forward-fill is
performed. Module 10 can decide how to handle missingness for econometric tests.

## Coverage Report

`merge_coverage_report.csv` contains one row per standard sector with:

- overlap window
- number of trading days
- return missing counts/ratios
- sentiment missing counts/ratios
- both-non-null counts/ratios
- article-day coverage
- total article count

## Local Run

```bash
python module9_merge_sentiment_return/merge_sentiment_return.py --price-wide VNStock/data/processed/daily_sector_price_wide.csv --sentiment-wide data/processed/daily_sector_sentiment_wide.csv --sentiment-long data/processed/daily_sector_sentiment_long.csv --output-wide data/processed/merged_sentiment_return_wide.csv --output-long data/processed/merged_sentiment_return_long.csv --coverage-report data/processed/merge_coverage_report.csv
```

Optional date bounds:

```bash
python module9_merge_sentiment_return/merge_sentiment_return.py --start-date 2026-01-01 --end-date 2026-06-30
```

## Missing Values

Missing news does not mean neutral sentiment. Missing sentiment is preserved as
`NaN`. Missing returns are also preserved as `NaN`.
