"""Map article publication timestamps to market sessions and trading dates."""

from __future__ import annotations

import argparse
import logging
import sys
from bisect import bisect_left, bisect_right
from datetime import date, time
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_ARTICLES_INPUT = Path("data/raw/articles.csv")
DEFAULT_TRADING_CALENDAR = Path("VNStock/data/processed/daily_sector_price_wide.csv")
DEFAULT_OUTPUT = Path("data/processed/articles_session_mapped.csv")
DEFAULT_TIMEZONE = "Asia/Ho_Chi_Minh"

ADDED_COLUMNS = [
    "published_at_vn",
    "session_bucket",
    "trade_date",
    "mapped_reason",
    "is_mapped",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module6_session_bucket")


def _normalize_date(value: Any) -> pd.Timestamp:
    """Convert a date-like value to a timezone-naive normalized Timestamp."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def load_trading_calendar(path: Path) -> list[pd.Timestamp]:
    """Load sorted unique trading dates from a Module 5 wide CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Trading calendar file not found: {path}")

    calendar_df = pd.read_csv(path)
    if "trade_date" not in calendar_df.columns:
        raise ValueError(f"Trading calendar is missing trade_date column: {path}")

    parsed_dates = pd.to_datetime(calendar_df["trade_date"], errors="coerce")
    invalid_count = int(parsed_dates.isna().sum())
    if invalid_count:
        logger.warning("Ignored %d invalid trade_date values in trading calendar", invalid_count)

    trading_days = sorted(
        {_normalize_date(value) for value in parsed_dates.dropna().tolist()}
    )
    if not trading_days:
        raise ValueError(f"Trading calendar contains no valid trade_date values: {path}")

    logger.info(
        "Loaded %d trading days: %s to %s",
        len(trading_days),
        trading_days[0].date(),
        trading_days[-1].date(),
    )
    return trading_days


def get_current_or_next_trading_day(
    value: date | pd.Timestamp, trading_days: list[pd.Timestamp]
) -> pd.Timestamp | None:
    """Return the trading day on or after the supplied date."""
    target = _normalize_date(value)
    index = bisect_left(trading_days, target)
    return trading_days[index] if index < len(trading_days) else None


def get_next_trading_day(
    value: date | pd.Timestamp, trading_days: list[pd.Timestamp]
) -> pd.Timestamp | None:
    """Return the first trading day strictly after the supplied date."""
    target = _normalize_date(value)
    index = bisect_right(trading_days, target)
    return trading_days[index] if index < len(trading_days) else None


def parse_published_at(value: Any, timezone: str) -> pd.Timestamp | None:
    """Parse a timestamp and convert or localize it to the requested timezone."""
    if value is None or pd.isna(value):
        return None

    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp):
            return None
        if timestamp.tzinfo is None:
            return timestamp.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
        return timestamp.tz_convert(timezone)
    except (TypeError, ValueError, OverflowError):
        return None


def _mapping_result(
    published_at_vn: pd.Timestamp | None,
    session_bucket: str,
    trade_date: pd.Timestamp | None,
    mapped_reason: str,
) -> dict[str, Any]:
    return {
        "published_at_vn": published_at_vn,
        "session_bucket": session_bucket,
        "trade_date": trade_date,
        "mapped_reason": mapped_reason,
        "is_mapped": trade_date is not None,
    }


def map_article_to_session(
    published_at: Any, trading_days: list[pd.Timestamp], timezone: str
) -> dict[str, Any]:
    """Map one article timestamp to a session bucket and valid trading date."""
    published_at_vn = parse_published_at(published_at, timezone)
    if published_at_vn is None:
        return _mapping_result(None, "UNKNOWN", None, "invalid_published_at")

    published_date = published_at_vn.tz_localize(None).normalize()
    is_trading_day = published_date in trading_days

    if not is_trading_day:
        next_day = get_current_or_next_trading_day(published_date, trading_days)
        reason = (
            "non_trading_day_to_next_trading_day"
            if next_day is not None
            else "no_next_trading_day_found"
        )
        if next_day is None:
            logger.warning("No next trading day for article published at %s", published_at_vn)
        return _mapping_result(published_at_vn, "NEXT_MORNING", next_day, reason)

    article_time = published_at_vn.time().replace(tzinfo=None)
    if article_time < time(9, 0):
        return _mapping_result(
            published_at_vn, "PRE_MARKET", published_date, "before_market_open"
        )
    if article_time < time(11, 30):
        return _mapping_result(
            published_at_vn, "MORNING", published_date, "during_morning_session"
        )
    if article_time < time(14, 45):
        return _mapping_result(
            published_at_vn, "AFTERNOON", published_date, "during_afternoon_session"
        )

    next_day = get_next_trading_day(published_date, trading_days)
    reason = "after_market_close" if next_day is not None else "no_next_trading_day_found"
    if next_day is None:
        logger.warning("No next trading day for article published at %s", published_at_vn)
    return _mapping_result(published_at_vn, "NEXT_MORNING", next_day, reason)


def map_articles_df(
    df: pd.DataFrame, trading_days: list[pd.Timestamp], timezone: str
) -> pd.DataFrame:
    """Map all articles while preserving every original input column."""
    if "published_at" not in df.columns:
        raise ValueError("Input articles CSV is missing required column: published_at")

    mapped = df["published_at"].apply(
        lambda value: map_article_to_session(value, trading_days, timezone)
    )
    mapping_df = pd.DataFrame(mapped.tolist(), index=df.index)
    result = df.copy()
    for column in ADDED_COLUMNS:
        result[column] = mapping_df[column]
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.date
    return result


def validate_output(df: pd.DataFrame) -> None:
    """Validate output columns and log mapping quality statistics."""
    missing_columns = [column for column in ADDED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Output is missing required columns: {missing_columns}")

    mapped_count = int(df["is_mapped"].fillna(False).astype(bool).sum())
    unmapped_count = len(df) - mapped_count
    null_trade_dates = int(df["trade_date"].isna().sum())

    logger.info("Input articles: %d", len(df))
    logger.info("Successfully mapped articles: %d", mapped_count)
    logger.info("Unmapped articles: %d", unmapped_count)
    logger.info("Session bucket distribution:\n%s", df["session_bucket"].value_counts(dropna=False))
    logger.info("Mapped reason distribution:\n%s", df["mapped_reason"].value_counts(dropna=False))
    valid_trade_dates = pd.to_datetime(df["trade_date"], errors="coerce").dropna()
    if valid_trade_dates.empty:
        logger.info("Trade date range: no mapped trade dates")
    else:
        logger.info(
            "Trade date range: %s to %s",
            valid_trade_dates.min().date(),
            valid_trade_dates.max().date(),
        )
    logger.info("Null trade_date values: %d", null_trade_dates)
    if null_trade_dates:
        logger.warning("Output contains %d null trade_date values", null_trade_dates)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--articles-input", type=Path, default=DEFAULT_ARTICLES_INPUT)
    parser.add_argument("--trading-calendar", type=Path, default=DEFAULT_TRADING_CALENDAR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--timezone", default=DEFAULT_TIMEZONE)
    return parser.parse_args()


def main() -> int:
    """Run the Module 6 CSV mapping pipeline."""
    args = parse_args()
    if not args.articles_input.exists():
        logger.error("Input articles file not found: %s", args.articles_input.as_posix())
        return 1

    try:
        trading_days = load_trading_calendar(args.trading_calendar)
        articles_df = pd.read_csv(args.articles_input)
        mapped_df = map_articles_df(articles_df, trading_days, args.timezone)
        validate_output(mapped_df)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        mapped_df.to_csv(args.output, index=False, date_format="%Y-%m-%d")
        logger.info("Output written to: %s", args.output)
        return 0
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
