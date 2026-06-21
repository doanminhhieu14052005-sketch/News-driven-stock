"""Aggregate article-sector CafeF sentiment into daily sector time series."""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/cafef_article_sector_long.csv")
DEFAULT_OUTPUT_LONG = Path("data/processed/daily_sector_sentiment_long.csv")
DEFAULT_OUTPUT_WIDE = Path("data/processed/daily_sector_sentiment_wide.csv")
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

# Vùng "trung tính" quanh 0 để phân loại pos/neu/neg khi sentiment_score là số liên tục.
NEUTRAL_EPS = 0.05

LONG_COLUMNS = [
    "trade_date",
    "sector_label",
    "article_count",
    "row_count",
    "weighted_article_count",
    "sentiment_mean",
    "sentiment_sum",
    "weighted_sentiment_sum",
    "sentiment_weighted_mean",
    "positive_count",
    "neutral_count",
    "negative_count",
    "positive_weighted_count",
    "neutral_weighted_count",
    "negative_weighted_count",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module8_daily_sentiment")


def read_input(path: Path) -> pd.DataFrame:
    """Read the Module 7 long CSV with Vietnamese-safe encoding."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_output(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def prepare_input(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clean and filter input rows before daily-sector aggregation."""
    required_columns = {"trade_date", "sector_label", "sentiment_score"}
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise RuntimeError(f"Input is missing required columns: {missing_columns}")

    prepared = df.copy()
    if "article_id" not in prepared.columns:
        logger.warning("Input is missing article_id; using row index as article key")
        prepared["article_id"] = pd.NA
    if "sector_weight" not in prepared.columns:
        logger.warning("Input is missing sector_weight; defaulting to 1.0")
        prepared["sector_weight"] = 1.0

    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    invalid_trade_date = int(prepared["trade_date"].isna().sum())
    if invalid_trade_date:
        logger.warning("Dropping %d rows with invalid trade_date", invalid_trade_date)
    prepared = prepared.dropna(subset=["trade_date"]).copy()
    prepared["trade_date"] = prepared["trade_date"].dt.normalize()

    prepared["sector_weight"] = pd.to_numeric(
        prepared["sector_weight"], errors="coerce"
    ).fillna(1.0)
    prepared["sentiment_score"] = pd.to_numeric(
        prepared["sentiment_score"], errors="coerce"
    )
    score_series = prepared["sentiment_score"].dropna()
    out_of_range = score_series[(score_series < -1.0) | (score_series > 1.0)]
    if len(out_of_range):
        raise RuntimeError(
            f"sentiment_score ngoài [-1, 1]: {sorted(out_of_range.unique().tolist())[:5]}"
        )

    non_standard_sector = int((~prepared["sector_label"].isin(STANDARD_SECTORS)).sum())
    if non_standard_sector:
        logger.warning("Dropping %d rows outside the 10 standard sectors", non_standard_sector)
    prepared = prepared.loc[prepared["sector_label"].isin(STANDARD_SECTORS)].copy()

    missing_article_id = prepared["article_id"].isna()
    prepared["_article_key"] = prepared["article_id"].astype(str)
    prepared.loc[missing_article_id, "_article_key"] = (
        "__row_" + prepared.index[missing_article_id].astype(str)
    )
    prepared["_weighted_sentiment"] = prepared["sentiment_score"] * prepared["sector_weight"]

    stats = {
        "input_rows": len(df),
        "valid_rows": len(prepared),
        "invalid_trade_date_rows": invalid_trade_date,
        "non_standard_sector_rows": non_standard_sector,
    }
    return prepared, stats


def _aggregate_group(group: pd.DataFrame) -> pd.Series:
    valid_sentiment = group["sentiment_score"].notna()
    valid_weight_sum = group.loc[valid_sentiment, "sector_weight"].sum()
    weighted_sentiment_sum = group.loc[valid_sentiment, "_weighted_sentiment"].sum(
        min_count=1
    )

    return pd.Series(
        {
            "article_count": group["_article_key"].nunique(),
            "row_count": len(group),
            "weighted_article_count": group["sector_weight"].sum(),
            "sentiment_mean": group["sentiment_score"].mean(),
            "sentiment_sum": group["sentiment_score"].sum(min_count=1),
            "weighted_sentiment_sum": weighted_sentiment_sum,
            "sentiment_weighted_mean": (
                weighted_sentiment_sum / valid_weight_sum if valid_weight_sum else np.nan
            ),
            "positive_count": int((group["sentiment_score"] > NEUTRAL_EPS).sum()),
            "neutral_count": int((group["sentiment_score"].abs() <= NEUTRAL_EPS).sum()),
            "negative_count": int((group["sentiment_score"] < -NEUTRAL_EPS).sum()),
            "positive_weighted_count": group.loc[
                group["sentiment_score"] > NEUTRAL_EPS, "sector_weight"
            ].sum(),
            "neutral_weighted_count": group.loc[
                group["sentiment_score"].abs() <= NEUTRAL_EPS, "sector_weight"
            ].sum(),
            "negative_weighted_count": group.loc[
                group["sentiment_score"] < -NEUTRAL_EPS, "sector_weight"
            ].sum(),
        }
    )


def aggregate_daily_sector_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate valid article-sector rows to daily-sector metrics."""
    if df.empty:
        return pd.DataFrame(columns=LONG_COLUMNS)

    records: list[dict[str, Any]] = []
    for (trade_date, sector_label), group in df.groupby(
        ["trade_date", "sector_label"], sort=True, dropna=False
    ):
        metrics = _aggregate_group(group).to_dict()
        records.append(
            {
                "trade_date": trade_date,
                "sector_label": sector_label,
                **metrics,
            }
        )

    grouped = pd.DataFrame(records)
    count_columns = [
        "article_count",
        "row_count",
        "positive_count",
        "neutral_count",
        "negative_count",
    ]
    for column in count_columns:
        grouped[column] = grouped[column].astype(int)
    grouped["trade_date"] = pd.to_datetime(grouped["trade_date"]).dt.strftime("%Y-%m-%d")
    return grouped[LONG_COLUMNS]


def build_wide_output(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivot sentiment_weighted_mean into wide daily sector format."""
    if long_df.empty:
        return pd.DataFrame(columns=["trade_date", *STANDARD_SECTORS])

    wide = long_df.pivot(
        index="trade_date", columns="sector_label", values="sentiment_weighted_mean"
    )
    wide = wide.reindex(columns=STANDARD_SECTORS)
    wide = wide.reset_index()
    return wide[["trade_date", *STANDARD_SECTORS]]


def validate_outputs(long_df: pd.DataFrame, wide_df: pd.DataFrame) -> None:
    """Validate schemas and sector/date constraints."""
    invalid_sectors = sorted(set(long_df["sector_label"].dropna()) - set(STANDARD_SECTORS))
    if invalid_sectors:
        raise RuntimeError(f"Long output contains invalid sectors: {invalid_sectors}")
    forbidden = {"MarketMacro", "Unmapped"}.intersection(set(long_df["sector_label"].dropna()))
    if forbidden:
        raise RuntimeError(f"Long output contains non-sector labels: {sorted(forbidden)}")

    missing_wide_columns = [sector for sector in STANDARD_SECTORS if sector not in wide_df.columns]
    if missing_wide_columns:
        raise RuntimeError(f"Wide output is missing sector columns: {missing_wide_columns}")

    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    for name, frame in {"long": long_df, "wide": wide_df}.items():
        if not frame.empty and not frame["trade_date"].astype(str).map(date_pattern.match).all():
            raise RuntimeError(f"{name} output contains non-YYYY-MM-DD trade_date values")


def print_summary(
    input_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    long_df: pd.DataFrame,
    wide_df: pd.DataFrame,
    stats: dict[str, int],
) -> None:
    print("input shape:", input_df.shape)
    print("number of valid rows used:", stats["valid_rows"])
    print("number of rows dropped because invalid trade_date:", stats["invalid_trade_date_rows"])
    print("long output shape:", long_df.shape)
    print("wide output shape:", wide_df.shape)
    if wide_df.empty:
        print("date range: no valid dates")
    else:
        print("date range:", wide_df["trade_date"].min(), "->", wide_df["trade_date"].max())
    print("sector coverage:", sorted(long_df["sector_label"].unique().tolist()))
    missing_ratio = wide_df.drop(columns=["trade_date"]).isna().mean()
    print("missing ratio in wide output:")
    print(missing_ratio.to_string())
    print("article_count by sector:")
    print(long_df.groupby("sector_label")["article_count"].sum().to_string())
    print("sentiment_weighted_mean summary by sector:")
    print(
        long_df.groupby("sector_label")["sentiment_weighted_mean"]
        .describe()
        .to_string()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-long", type=Path, default=DEFAULT_OUTPUT_LONG)
    parser.add_argument("--output-wide", type=Path, default=DEFAULT_OUTPUT_WIDE)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_df = read_input(args.input)
        valid_df, stats = prepare_input(input_df)
        long_df = aggregate_daily_sector_sentiment(valid_df)
        wide_df = build_wide_output(long_df)
        validate_outputs(long_df, wide_df)
        write_output(long_df, args.output_long)
        write_output(wide_df, args.output_wide)
        print_summary(input_df, valid_df, long_df, wide_df, stats)
        print("long output written to:", args.output_long.as_posix())
        print("wide output written to:", args.output_wide.as_posix())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
