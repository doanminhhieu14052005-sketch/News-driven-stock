"""Fetch daily OHLCV data and build equal-weighted sector returns."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


START_DATE = "2024-01-01"
END_DATE = "2024-12-31"
INTERVAL = "1D"
DEFAULT_OUTPUT_DIR = Path("data/processed")

SECTOR_TICKERS = {
    "Banking": ["VCB", "BID", "CTG"],
    "RealEstate": ["VIC", "VHM", "NVL"],
    "SteelMaterials": ["HPG", "HSG", "NKG"],
    "Technology": ["FPT", "CMG", "VGI"],
    "ConsumerStaples": ["MSN", "SAB", "VNM"],
    "ConsumerDiscretionary": ["MWG", "PNJ", "DGW"],
    "Energy": ["GAS", "PLX", "PVD"],
    "IndustrialLogistics": ["GMD", "HAH", "ACV"],
    "Healthcare": ["DHG", "IMP", "DMC"],
    "Utilities": ["POW", "NT2", "REE"],
}

RETRY_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 2.0
REQUEST_DELAY_SECONDS = 0.5
BATCH_SIZE = 18
BATCH_DELAY_SECONDS = 65.0
OUTPUT_COLUMNS = ["trade_date", "ticker", "open", "high", "low", "close", "volume"]

for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module5_vnstock_price")


def _call_vnstock_api(
    ticker: str, start_date: str, end_date: str, interval: str
) -> pd.DataFrame:
    """Call the first vnstock API available in the installed version."""
    errors: list[str] = []

    try:
        from vnstock.api.quote import Quote

        return Quote(symbol=ticker, source="VCI").history(
            start=start_date, end=end_date, interval=interval
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"vnstock.api.quote.Quote: {exc}")

    try:
        from vnstock import Vnstock

        stock = Vnstock().stock(symbol=ticker, source="VCI")
        return stock.quote.history(start=start_date, end=end_date, interval=interval)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"vnstock.Vnstock: {exc}")

    try:
        from vnstock import stock_historical_data

        return stock_historical_data(
            symbol=ticker,
            start_date=start_date,
            end_date=end_date,
            resolution=interval,
            type="stock",
            beautify=True,
            decor=False,
            source="DNSE",
        )
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        errors.append(f"vnstock.stock_historical_data: {exc}")

    raise RuntimeError("No compatible vnstock API found. " + " | ".join(errors))


def get_price_one_ticker(
    ticker: str, start_date: str, end_date: str, interval: str = INTERVAL
) -> pd.DataFrame:
    """Fetch one ticker with retries and vnstock-version fallbacks."""
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        try:
            result = _call_vnstock_api(ticker, start_date, end_date, interval)
            if result is None:
                return pd.DataFrame()
            return result
        except Exception as exc:
            logger.warning(
                "%s: fetch attempt %d/%d failed: %s",
                ticker,
                attempt,
                RETRY_ATTEMPTS,
                exc,
            )
            if attempt < RETRY_ATTEMPTS:
                time.sleep(RETRY_DELAY_SECONDS)
    return pd.DataFrame()


def _canonical_column_name(name: Any) -> str:
    return "".join(character for character in str(name).lower() if character.isalnum())


def normalize_ohlcv_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Normalize a vnstock result to the required daily OHLCV schema."""
    if df is None or df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    aliases = {
        "trade_date": {"time", "date", "tradingdate", "trading_date", "tradedate"},
        "open": {"open", "openprice", "open_price"},
        "high": {"high", "highprice", "high_price"},
        "low": {"low", "lowprice", "low_price"},
        "close": {"close", "closeprice", "close_price"},
        "volume": {"volume", "vol", "totalvolume", "total_volume"},
    }
    canonical_aliases = {
        target: {_canonical_column_name(alias) for alias in values}
        for target, values in aliases.items()
    }
    rename_map: dict[Any, str] = {}
    for column in df.columns:
        canonical = _canonical_column_name(column)
        for target, accepted in canonical_aliases.items():
            if canonical in accepted:
                rename_map[column] = target
                break

    normalized = df.rename(columns=rename_map).copy()
    if "trade_date" not in normalized.columns:
        logger.warning("%s: missing date column; ticker skipped", ticker)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    if "close" not in normalized.columns:
        logger.warning("%s: missing close column; ticker skipped", ticker)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    for column in ["open", "high", "low", "volume"]:
        if column not in normalized.columns:
            logger.warning("%s: missing %s column; values set to NaN", ticker, column)
            normalized[column] = pd.NA

    normalized["trade_date"] = pd.to_datetime(
        normalized["trade_date"], errors="coerce"
    ).dt.normalize()
    for column in ["open", "high", "low", "close", "volume"]:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized["ticker"] = ticker

    normalized = normalized.dropna(subset=["trade_date", "close"])
    normalized = normalized[OUTPUT_COLUMNS]
    normalized = normalized.drop_duplicates(
        subset=["ticker", "trade_date"], keep="last"
    )
    return normalized.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def fetch_all_prices(
    sector_tickers: dict[str, list[str]],
    start_date: str,
    end_date: str,
    interval: str = INTERVAL,
) -> pd.DataFrame:
    """Fetch and normalize all configured ticker prices."""
    frames: list[pd.DataFrame] = []
    successful: list[str] = []
    failed: list[str] = []
    tickers = list(dict.fromkeys(t for values in sector_tickers.values() for t in values))

    for ticker_number, ticker in enumerate(tickers, start=1):
        if ticker_number > 1 and (ticker_number - 1) % BATCH_SIZE == 0:
            logger.info(
                "Processed %d tickers; sleeping %.0f seconds to avoid rate limits",
                ticker_number - 1,
                BATCH_DELAY_SECONDS,
            )
            time.sleep(BATCH_DELAY_SECONDS)
        logger.info("Downloading ticker %s", ticker)
        raw = get_price_one_ticker(ticker, start_date, end_date, interval)
        normalized = normalize_ohlcv_df(raw, ticker)
        if not normalized.empty:
            start_timestamp = pd.Timestamp(start_date).normalize()
            end_timestamp = pd.Timestamp(end_date).normalize()
            normalized = normalized.loc[
                normalized["trade_date"].between(start_timestamp, end_timestamp)
            ].reset_index(drop=True)
        if normalized.empty:
            failed.append(ticker)
            logger.warning("%s: no valid OHLCV data", ticker)
        else:
            frames.append(normalized)
            successful.append(ticker)
            logger.info("%s: downloaded %d valid rows", ticker, len(normalized))
        time.sleep(REQUEST_DELAY_SECONDS)

    logger.info("Successful tickers (%d/%d): %s", len(successful), len(tickers), successful)
    logger.info("Failed tickers (%d/%d): %s", len(failed), len(tickers), failed)
    if not frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    prices = pd.concat(frames, ignore_index=True)
    prices = prices.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    logger.info("Raw normalized price dataframe shape: %s", prices.shape)
    logger.info(
        "Actual price date range: %s to %s",
        prices["trade_date"].min().date(),
        prices["trade_date"].max().date(),
    )
    return prices


def compute_ticker_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """Compute close-to-close daily return separately for every ticker."""
    if price_df.empty:
        return price_df.assign(ticker_return=pd.Series(dtype="float64"))

    returns = price_df.sort_values(["ticker", "trade_date"]).copy()
    returns["ticker_return"] = returns.groupby("ticker", sort=False)["close"].transform(
        lambda values: values.pct_change(fill_method=None)
    )
    return returns


def compute_sector_returns(
    return_df: pd.DataFrame, sector_tickers: dict[str, list[str]]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build equal-weighted daily sector returns in wide and long formats."""
    sectors = list(sector_tickers)
    long_columns = [
        "trade_date",
        "sector",
        "sector_return",
        "ticker_count",
        "missing_count",
    ]
    if return_df.empty:
        return (
            pd.DataFrame(columns=["trade_date", *sectors]),
            pd.DataFrame(columns=long_columns),
        )

    ticker_returns = return_df.pivot_table(
        index="trade_date", columns="ticker", values="ticker_return", aggfunc="last"
    ).sort_index()
    wide = pd.DataFrame(index=ticker_returns.index)
    long_frames: list[pd.DataFrame] = []

    for sector, configured_tickers in sector_tickers.items():
        available = [ticker for ticker in configured_tickers if ticker in ticker_returns]
        sector_values = ticker_returns.reindex(columns=available)
        wide[sector] = sector_values.mean(axis=1, skipna=True)
        counts = sector_values.notna().sum(axis=1)
        long_frames.append(
            pd.DataFrame(
                {
                    "trade_date": ticker_returns.index,
                    "sector": sector,
                    "sector_return": wide[sector],
                    "ticker_count": counts,
                    "missing_count": len(configured_tickers) - counts,
                }
            )
        )

    wide = wide.reset_index()
    all_missing = wide[sectors].isna().all(axis=1)
    removed_dates = set(wide.loc[all_missing, "trade_date"])
    wide = wide.loc[~all_missing].reset_index(drop=True)
    long_df = pd.concat(long_frames, ignore_index=True)
    if removed_dates:
        long_df = long_df.loc[~long_df["trade_date"].isin(removed_dates)].reset_index(
            drop=True
        )
    return wide, long_df[long_columns]


def save_outputs(
    wide_df: pd.DataFrame, long_df: pd.DataFrame, output_dir: str | Path
) -> tuple[Path, Path]:
    """Save required wide and long CSV outputs."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    wide_path = output_path / "daily_sector_price_wide.csv"
    long_path = output_path / "daily_sector_price_long.csv"
    wide_df.to_csv(wide_path, index=False, date_format="%Y-%m-%d")
    long_df.to_csv(long_path, index=False, date_format="%Y-%m-%d")
    logger.info("Exported wide CSV: %s", wide_path)
    logger.info("Exported long CSV: %s", long_path)
    return wide_path, long_path


def write_to_mongo(
    long_df: pd.DataFrame,
    mongo_uri: str | None,
    mongo_db: str,
    collection_name: str = "daily_sector_price",
) -> None:
    """Upsert long-format sector returns to MongoDB."""
    if not mongo_uri:
        logger.warning("MongoDB write requested but mongo_uri is missing; skipping")
        return
    if long_df.empty:
        logger.warning("MongoDB write requested but long output is empty; skipping")
        return

    try:
        from pymongo import MongoClient, UpdateOne
    except ImportError:
        logger.warning("pymongo is not installed; skipping MongoDB write")
        return

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
    try:
        collection = client[mongo_db][collection_name]
        operations = []
        for record in long_df.to_dict("records"):
            record["trade_date"] = pd.Timestamp(record["trade_date"]).to_pydatetime()
            for key, value in list(record.items()):
                if pd.isna(value):
                    record[key] = None
                elif hasattr(value, "item"):
                    record[key] = value.item()
            operations.append(
                UpdateOne(
                    {"trade_date": record["trade_date"], "sector": record["sector"]},
                    {"$set": record},
                    upsert=True,
                )
            )
        result = collection.bulk_write(operations, ordered=False)
        logger.info(
            "MongoDB write complete: %d upserted, %d modified",
            result.upserted_count,
            result.modified_count,
        )
    except Exception as exc:
        logger.warning("MongoDB write failed; CSV outputs remain available: %s", exc)
    finally:
        client.close()


def validate_outputs(wide_df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    """Validate output schemas and log data-quality warnings."""
    sectors = list(SECTOR_TICKERS)
    expected_long = {
        "trade_date",
        "sector",
        "sector_return",
        "ticker_count",
        "missing_count",
    }
    errors: list[str] = []
    if "trade_date" not in wide_df:
        errors.append("wide_df is missing trade_date")
    missing_sectors = [sector for sector in sectors if sector not in wide_df]
    if missing_sectors:
        errors.append(f"wide_df is missing sectors: {missing_sectors}")
    missing_long = expected_long.difference(long_df.columns)
    if missing_long:
        errors.append(f"long_df is missing columns: {sorted(missing_long)}")
    if "trade_date" in wide_df and wide_df["trade_date"].duplicated().any():
        errors.append("wide_df contains duplicate trade_date values")
    if not long_df.empty and long_df.groupby("trade_date").size().max() > len(sectors):
        errors.append("long_df contains more than 10 sectors for at least one date")
    if "sector_return" in long_df and not pd.api.types.is_numeric_dtype(
        long_df["sector_return"]
    ):
        errors.append("long_df sector_return is not numeric")
    if errors:
        raise ValueError("Output validation failed: " + "; ".join(errors))

    logger.info("Wide output rows: %d; sector columns: %d", len(wide_df), len(sectors))
    if len(wide_df) < 200:
        logger.warning("Only %d trading days found; expected at least 200", len(wide_df))
    for sector in sectors:
        missing_ratio = float(wide_df[sector].isna().mean()) if len(wide_df) else 1.0
        logger.info("%s missing ratio: %.2f%%", sector, missing_ratio * 100)
        if missing_ratio > 0.20:
            logger.warning("%s missing ratio exceeds 20%%", sector)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", "--start", default=START_DATE)
    parser.add_argument("--end-date", "--end", default=END_DATE)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--write-mongo", action="store_true")
    parser.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"))
    parser.add_argument("--mongo-db", default=os.getenv("MONGO_DB_NAME", "news_spillover_hose"))
    return parser.parse_args()


def main() -> None:
    """Run the complete price collection and sector-return pipeline."""
    args = parse_args()
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if start > end:
        raise ValueError("start_date must be on or before end_date")

    logger.info("Starting vnstock sector price module")
    logger.info(
        "start_date=%s end_date=%s output_dir=%s",
        args.start_date,
        args.end_date,
        args.output_dir,
    )
    prices = fetch_all_prices(SECTOR_TICKERS, args.start_date, args.end_date)
    returns = compute_ticker_returns(prices)
    wide_df, long_df = compute_sector_returns(returns, SECTOR_TICKERS)
    validate_outputs(wide_df, long_df)
    save_outputs(wide_df, long_df, args.output_dir)
    if args.write_mongo:
        write_to_mongo(long_df, args.mongo_uri, args.mongo_db)
    logger.info("Module completed")


if __name__ == "__main__":
    main()
