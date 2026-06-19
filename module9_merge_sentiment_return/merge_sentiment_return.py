"""Merge daily sector sentiment with daily sector returns."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_PRICE_WIDE = Path("VNStock/data/processed/daily_sector_price_wide.csv")
DEFAULT_SENTIMENT_WIDE = Path("data/processed/daily_sector_sentiment_wide.csv")
DEFAULT_SENTIMENT_LONG = Path("data/processed/daily_sector_sentiment_long.csv")
DEFAULT_OUTPUT_WIDE = Path("data/processed/merged_sentiment_return_wide.csv")
DEFAULT_OUTPUT_LONG = Path("data/processed/merged_sentiment_return_long.csv")
DEFAULT_COVERAGE_REPORT = Path("data/processed/merge_coverage_report.csv")
CSV_ENCODING = "utf-8-sig"

STANDARD_SECTORS = [
    "Banking",
    "RealEstate",
    "SteelMaterials",
    "Technology",
    "ConsumerStaples",
    "ConsumerDiscretionary",
    "Energy",
    "IndustrialLogistics",
    "Healthcare",
    "Utilities",
]

SENTIMENT_METRICS = [
    "sentiment_weighted_mean",
    "sentiment_mean",
    "sentiment_sum",
    "weighted_sentiment_sum",
    "article_count",
    "row_count",
    "weighted_article_count",
    "positive_count",
    "neutral_count",
    "negative_count",
    "positive_weighted_count",
    "neutral_weighted_count",
    "negative_weighted_count",
]

LONG_COLUMNS = [
    "trade_date",
    "sector_label",
    "sector_return",
    *SENTIMENT_METRICS,
]

REPORT_COLUMNS = [
    "sector_label",
    "overlap_start",
    "overlap_end",
    "n_trading_days",
    "return_non_null_count",
    "return_missing_count",
    "return_missing_ratio",
    "sentiment_non_null_count",
    "sentiment_missing_count",
    "sentiment_missing_ratio",
    "both_non_null_count",
    "both_non_null_ratio",
    "article_days_count",
    "total_article_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module9_merge_sentiment_return")


def read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV using the repository's Vietnamese-safe encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def _parse_date(value: str | None, arg_name: str) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise RuntimeError(f"Invalid {arg_name}: {value}")
    return pd.Timestamp(parsed).normalize()


def _prepare_wide(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    """Validate a wide daily sector table and normalize its date column."""
    if "trade_date" not in df.columns:
        raise RuntimeError(f"{table_name} is missing trade_date column")

    missing_sectors = [sector for sector in STANDARD_SECTORS if sector not in df.columns]
    if missing_sectors:
        raise RuntimeError(f"{table_name} is missing sector columns: {missing_sectors}")

    prepared = df[["trade_date", *STANDARD_SECTORS]].copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    invalid_dates = int(prepared["trade_date"].isna().sum())
    if invalid_dates:
        logger.warning("%s: dropping %d rows with invalid trade_date", table_name, invalid_dates)
    prepared = prepared.dropna(subset=["trade_date"]).copy()
    prepared["trade_date"] = prepared["trade_date"].dt.normalize()
    for sector in STANDARD_SECTORS:
        prepared[sector] = pd.to_numeric(prepared[sector], errors="coerce")

    duplicate_dates = int(prepared["trade_date"].duplicated().sum())
    if duplicate_dates:
        logger.warning("%s: dropping %d duplicate trade_date rows", table_name, duplicate_dates)
        prepared = prepared.drop_duplicates(subset=["trade_date"], keep="last")

    return prepared.sort_values("trade_date").reset_index(drop=True)


def _prepare_sentiment_long(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the Module 8 long sentiment table."""
    required = {"trade_date", "sector_label", *SENTIMENT_METRICS}
    missing_columns = sorted(required - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"sentiment long is missing columns: {missing_columns}")

    prepared = df[["trade_date", "sector_label", *SENTIMENT_METRICS]].copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    invalid_dates = int(prepared["trade_date"].isna().sum())
    if invalid_dates:
        logger.warning("sentiment long: dropping %d rows with invalid trade_date", invalid_dates)
    prepared = prepared.dropna(subset=["trade_date"]).copy()
    prepared["trade_date"] = prepared["trade_date"].dt.normalize()

    invalid_sectors = sorted(set(prepared["sector_label"].dropna()) - set(STANDARD_SECTORS))
    if invalid_sectors:
        logger.warning("sentiment long: dropping non-standard sectors: %s", invalid_sectors)
    prepared = prepared.loc[prepared["sector_label"].isin(STANDARD_SECTORS)].copy()

    for column in SENTIMENT_METRICS:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    duplicate_keys = int(prepared.duplicated(subset=["trade_date", "sector_label"]).sum())
    if duplicate_keys:
        logger.warning(
            "sentiment long: dropping %d duplicate trade_date + sector_label rows",
            duplicate_keys,
        )
        prepared = prepared.drop_duplicates(
            subset=["trade_date", "sector_label"], keep="last"
        )
    return prepared.sort_values(["trade_date", "sector_label"]).reset_index(drop=True)


def determine_overlap(
    price_wide: pd.DataFrame,
    sentiment_wide: pd.DataFrame,
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Compute the common date window, with optional user bounds."""
    price_start, price_end = price_wide["trade_date"].min(), price_wide["trade_date"].max()
    sentiment_start = sentiment_wide["trade_date"].min()
    sentiment_end = sentiment_wide["trade_date"].max()
    overlap_start = max(price_start, sentiment_start)
    overlap_end = min(price_end, sentiment_end)

    user_start = _parse_date(start_date, "start-date")
    user_end = _parse_date(end_date, "end-date")
    if user_start is not None:
        overlap_start = max(overlap_start, user_start)
    if user_end is not None:
        overlap_end = min(overlap_end, user_end)

    if overlap_start > overlap_end:
        raise RuntimeError(
            "No overlap date window between price and sentiment inputs "
            f"after bounds: {overlap_start.date()} > {overlap_end.date()}"
        )
    return overlap_start, overlap_end


def build_merged_wide(
    price_wide: pd.DataFrame,
    sentiment_wide: pd.DataFrame,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> pd.DataFrame:
    """Use the price calendar as base and left join daily sentiment."""
    price_base = price_wide.loc[
        price_wide["trade_date"].between(overlap_start, overlap_end)
    ].copy()
    sentiment_base = sentiment_wide.loc[
        sentiment_wide["trade_date"].between(overlap_start, overlap_end)
    ].copy()

    price_base = price_base.rename(
        columns={sector: f"ret_{sector}" for sector in STANDARD_SECTORS}
    )
    sentiment_base = sentiment_base.rename(
        columns={sector: f"sent_{sector}" for sector in STANDARD_SECTORS}
    )
    merged = price_base[["trade_date", *[f"ret_{s}" for s in STANDARD_SECTORS]]].merge(
        sentiment_base[["trade_date", *[f"sent_{s}" for s in STANDARD_SECTORS]]],
        on="trade_date",
        how="left",
    )
    merged["trade_date"] = merged["trade_date"].dt.strftime("%Y-%m-%d")
    return merged[
        [
            "trade_date",
            *[f"ret_{sector}" for sector in STANDARD_SECTORS],
            *[f"sent_{sector}" for sector in STANDARD_SECTORS],
        ]
    ]


def build_merged_long(
    price_wide: pd.DataFrame,
    sentiment_long: pd.DataFrame,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> pd.DataFrame:
    """Create one row per trading day and standard sector."""
    price_base = price_wide.loc[
        price_wide["trade_date"].between(overlap_start, overlap_end)
    ].copy()
    price_long = price_base.melt(
        id_vars=["trade_date"],
        value_vars=STANDARD_SECTORS,
        var_name="sector_label",
        value_name="sector_return",
    )
    price_long = price_long.sort_values(["trade_date", "sector_label"]).reset_index(drop=True)

    sentiment_base = sentiment_long.loc[
        sentiment_long["trade_date"].between(overlap_start, overlap_end)
    ].copy()
    merged = price_long.merge(
        sentiment_base[["trade_date", "sector_label", *SENTIMENT_METRICS]],
        on=["trade_date", "sector_label"],
        how="left",
    )
    merged["trade_date"] = merged["trade_date"].dt.strftime("%Y-%m-%d")
    return merged[LONG_COLUMNS]


def build_coverage_report(
    merged_long: pd.DataFrame,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> pd.DataFrame:
    """Build sector-level coverage diagnostics for the merged dataset."""
    reports: list[dict[str, Any]] = []
    for sector in STANDARD_SECTORS:
        sector_df = merged_long.loc[merged_long["sector_label"] == sector]
        n_days = len(sector_df)
        return_non_null = int(sector_df["sector_return"].notna().sum())
        sentiment_non_null = int(sector_df["sentiment_weighted_mean"].notna().sum())
        both_non_null = int(
            (sector_df["sector_return"].notna() & sector_df["sentiment_weighted_mean"].notna()).sum()
        )
        article_days = int(sector_df["article_count"].notna().sum())
        total_article_count = sector_df["article_count"].sum(min_count=1)
        reports.append(
            {
                "sector_label": sector,
                "overlap_start": overlap_start.strftime("%Y-%m-%d"),
                "overlap_end": overlap_end.strftime("%Y-%m-%d"),
                "n_trading_days": n_days,
                "return_non_null_count": return_non_null,
                "return_missing_count": n_days - return_non_null,
                "return_missing_ratio": (n_days - return_non_null) / n_days if n_days else np.nan,
                "sentiment_non_null_count": sentiment_non_null,
                "sentiment_missing_count": n_days - sentiment_non_null,
                "sentiment_missing_ratio": (
                    (n_days - sentiment_non_null) / n_days if n_days else np.nan
                ),
                "both_non_null_count": both_non_null,
                "both_non_null_ratio": both_non_null / n_days if n_days else np.nan,
                "article_days_count": article_days,
                "total_article_count": total_article_count,
            }
        )
    return pd.DataFrame(reports)[REPORT_COLUMNS]


def validate_outputs(
    merged_wide: pd.DataFrame,
    merged_long: pd.DataFrame,
    coverage_report: pd.DataFrame,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> None:
    """Validate required schemas and date/sector constraints."""
    expected_wide_columns = [
        "trade_date",
        *[f"ret_{sector}" for sector in STANDARD_SECTORS],
        *[f"sent_{sector}" for sector in STANDARD_SECTORS],
    ]
    missing_wide = [column for column in expected_wide_columns if column not in merged_wide.columns]
    if missing_wide:
        raise RuntimeError(f"Merged wide output is missing columns: {missing_wide}")

    invalid_long_sectors = sorted(set(merged_long["sector_label"].dropna()) - set(STANDARD_SECTORS))
    if invalid_long_sectors:
        raise RuntimeError(f"Merged long output contains invalid sectors: {invalid_long_sectors}")
    forbidden = {"MarketMacro", "Unmapped"}.intersection(set(merged_long["sector_label"].dropna()))
    if forbidden:
        raise RuntimeError(f"Merged long output contains non-sector labels: {sorted(forbidden)}")

    long_dates = pd.to_datetime(merged_long["trade_date"], errors="coerce")
    wide_dates = pd.to_datetime(merged_wide["trade_date"], errors="coerce")
    if long_dates.isna().any() or wide_dates.isna().any():
        raise RuntimeError("Merged outputs contain invalid trade_date values")
    if wide_dates.min() < overlap_start or wide_dates.max() > overlap_end:
        raise RuntimeError("Merged wide date range is outside overlap window")
    if long_dates.min() < overlap_start or long_dates.max() > overlap_end:
        raise RuntimeError("Merged long date range is outside overlap window")

    if len(coverage_report) != len(STANDARD_SECTORS):
        raise RuntimeError("Coverage report must contain exactly 10 sector rows")
    missing_report_cols = [column for column in REPORT_COLUMNS if column not in coverage_report.columns]
    if missing_report_cols:
        raise RuntimeError(f"Coverage report is missing columns: {missing_report_cols}")


def print_summary(
    price_wide: pd.DataFrame,
    sentiment_wide: pd.DataFrame,
    sentiment_long: pd.DataFrame,
    merged_wide: pd.DataFrame,
    merged_long: pd.DataFrame,
    overlap_start: pd.Timestamp,
    overlap_end: pd.Timestamp,
) -> None:
    print("price shape:", price_wide.shape)
    print("sentiment wide shape:", sentiment_wide.shape)
    print("sentiment long shape:", sentiment_long.shape)
    print("merged wide shape:", merged_wide.shape)
    print("merged long shape:", merged_long.shape)
    print(
        "price date range:",
        price_wide["trade_date"].min().date(),
        "->",
        price_wide["trade_date"].max().date(),
    )
    print(
        "sentiment date range:",
        sentiment_wide["trade_date"].min().date(),
        "->",
        sentiment_wide["trade_date"].max().date(),
    )
    print("overlap date range:", overlap_start.date(), "->", overlap_end.date())
    print("number of trading days in overlap:", len(merged_wide))
    print("missing sentiment ratio by sector:")
    print(merged_wide[[f"sent_{s}" for s in STANDARD_SECTORS]].isna().mean().to_string())
    print("missing return ratio by sector:")
    print(merged_wide[[f"ret_{s}" for s in STANDARD_SECTORS]].isna().mean().to_string())
    print("first 5 rows of merged wide:")
    print(merged_wide.head().to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--price-wide", type=Path, default=DEFAULT_PRICE_WIDE)
    parser.add_argument("--sentiment-wide", type=Path, default=DEFAULT_SENTIMENT_WIDE)
    parser.add_argument("--sentiment-long", type=Path, default=DEFAULT_SENTIMENT_LONG)
    parser.add_argument("--output-wide", type=Path, default=DEFAULT_OUTPUT_WIDE)
    parser.add_argument("--output-long", type=Path, default=DEFAULT_OUTPUT_LONG)
    parser.add_argument("--coverage-report", type=Path, default=DEFAULT_COVERAGE_REPORT)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        price_raw = read_csv(args.price_wide)
        sentiment_wide_raw = read_csv(args.sentiment_wide)
        sentiment_long_raw = read_csv(args.sentiment_long)

        price_wide = _prepare_wide(price_raw, "price wide")
        sentiment_wide = _prepare_wide(sentiment_wide_raw, "sentiment wide")
        sentiment_long = _prepare_sentiment_long(sentiment_long_raw)
        overlap_start, overlap_end = determine_overlap(
            price_wide,
            sentiment_wide,
            args.start_date,
            args.end_date,
        )

        merged_wide = build_merged_wide(price_wide, sentiment_wide, overlap_start, overlap_end)
        merged_long = build_merged_long(price_wide, sentiment_long, overlap_start, overlap_end)
        coverage_report = build_coverage_report(merged_long, overlap_start, overlap_end)
        validate_outputs(merged_wide, merged_long, coverage_report, overlap_start, overlap_end)

        write_csv(merged_wide, args.output_wide)
        write_csv(merged_long, args.output_long)
        write_csv(coverage_report, args.coverage_report)
        print_summary(
            price_wide,
            sentiment_wide,
            sentiment_long,
            merged_wide,
            merged_long,
            overlap_start,
            overlap_end,
        )
        print("merged wide written to:", args.output_wide.as_posix())
        print("merged long written to:", args.output_long.as_posix())
        print("coverage report written to:", args.coverage_report.as_posix())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
