# Module 7 - CafeF Sector & Sentiment Mapper

## Purpose

Module 7 normalizes raw CafeF `impact`, `sector`, and ticker information into
analysis-ready sentiment and sector labels. It does not train or call a new
sentiment classifier. It converts the existing CafeF summary fields into:

- `sentiment_score`
- `sentiment_label`
- `sector_label`
- `sector_weight`
- mapping audit columns

## Input

Default input:

`data/processed/articles_session_mapped.csv`

The expected upstream input is produced by Module 6 and should include
`trade_date`, `published_at_vn`, `session_bucket`, and CafeF summary columns
such as `summary_json.impact`, `summary_json.sector`, and
`summary_json.tickers`.

## Outputs

Article-level output:

`data/processed/cafef_articles_labeled.csv`

This keeps every original input column and adds normalized sentiment and sector
mapping columns. `Unmapped` and `MarketMacro` rows are kept here for audit.

Long sector output:

`data/processed/cafef_article_sector_long.csv`

This has one row per article-sector and is intended for daily sector sentiment
aggregation. It excludes `Unmapped` and `MarketMacro` by default, so only the 10
standard project sectors are present.

Both CSV outputs are written with `utf-8-sig` for Vietnamese text compatibility
with Excel, GitHub preview, and VS Code.

## Mapping Rules

Sector mapping priority:

1. Valid ticker mapping using Module 5 ticker proxies.
2. Keyword mapping from `summary_json.sector`.
3. Macro detection for broad market or macroeconomic news.
4. Keyword fallback from title/category/summary.
5. `Unmapped` if no reliable sector signal exists.

Ticker mapping uses the same 10-sector proxy list as Module 5:

- Banking
- RealEstate
- SteelMaterials
- Technology
- ConsumerStaples
- ConsumerDiscretionary
- Energy
- IndustrialLogistics
- Healthcare
- Utilities

Macro news is labeled `MarketMacro` in the article-level file but is not used
in the 10-sector long output. Examples include interest rates, exchange rates,
CPI, GDP, tax, broad policy, and general stock-market news.

## Local Run

```bash
python module7_sector_sentiment/cafef_sector_sentiment_mapper.py --input data/processed/articles_session_mapped.csv --output data/processed/cafef_articles_labeled.csv --long-output data/processed/cafef_article_sector_long.csv
```

## Notes

Do not force missing sectors into Banking or any arbitrary sector. If there is
not enough information, keep the article as `Unmapped` for audit. If the news is
macro or broad market news, keep it as `MarketMacro` at article level and omit it
from sector-level aggregation.
