# Module 8 - Daily Sector Sentiment Aggregation

## Purpose

Module 8 converts article-sector sentiment rows from Module 7 into daily
sector-level sentiment time series. These outputs are intended for Granger
causality, lead-lag, and network comparison with Module 5 sector returns.

## Input

Default input:

`data/processed/cafef_article_sector_long.csv`

Expected columns include:

- `article_id`
- `trade_date`
- `sector_label`
- `sector_weight`
- `sentiment_score`
- `sentiment_label`

Only the 10 standard project sectors are aggregated. `MarketMacro` and
`Unmapped` are excluded.

## Outputs

Long daily-sector output:

`data/processed/daily_sector_sentiment_long.csv`

Wide daily-sector output:

`data/processed/daily_sector_sentiment_wide.csv`

The wide file has one row per `trade_date`, one column per sector, and values
equal to `sentiment_weighted_mean`. Missing news remains `NaN`; Module 8 does
not fill missing sentiment with zero and does not forward-fill.

Both CSV outputs are written with `utf-8-sig`.

## Aggregation Formulas

For each `trade_date` and `sector_label`:

- `article_count`: unique article count
- `row_count`: number of article-sector rows
- `weighted_article_count`: sum of `sector_weight`
- `sentiment_mean`: mean of `sentiment_score`
- `sentiment_sum`: sum of `sentiment_score`
- `weighted_sentiment_sum`: sum of `sentiment_score * sector_weight`
- `sentiment_weighted_mean`: weighted mean using only rows with valid sentiment
- `positive_count`, `neutral_count`, `negative_count`: row counts by score
- weighted sentiment counts: sum of `sector_weight` for each sentiment score

The weighted mean skips missing `sentiment_score` values so missing sentiment is
not treated as neutral.

## Local Run

```bash
python module8_daily_sentiment/daily_sector_sentiment_aggregator.py --input data/processed/cafef_article_sector_long.csv --output-long data/processed/daily_sector_sentiment_long.csv --output-wide data/processed/daily_sector_sentiment_wide.csv
```

## Notes

`sector_weight` prevents multi-sector articles from being counted too heavily.
For example, an article mapped to two sectors has weight `0.5` in each sector.
Daily-sector combinations without news are intentionally left as missing values.
