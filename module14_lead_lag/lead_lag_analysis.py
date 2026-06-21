"""RQ3 - Lead-lag cross-correlation between sector sentiment and sector return."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr


DEFAULT_INPUT = Path("data/processed/merged_sentiment_return_wide.csv")
DEFAULT_ALL_OUTPUT = Path("data/processed/lead_lag_all.csv")
DEFAULT_BEST_OUTPUT = Path("data/processed/lead_lag_best.csv")
CSV_ENCODING = "utf-8-sig"

MIN_OBS = 30

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

ALL_COLUMNS = [
    "s_sector",
    "r_sector",
    "lag",
    "corr",
    "pvalue",
    "significant",
    "n_obs",
]

BEST_COLUMNS = [
    "s_sector",
    "r_sector",
    "best_lag",
    "corr",
    "pvalue",
    "significant",
    "n_obs",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module14_lead_lag")


def read_input(path: Path) -> pd.DataFrame:
    """Read merged sentiment-return data."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def prepare_input(df: pd.DataFrame, fill_missing_sentiment: bool = True) -> pd.DataFrame:
    """Validate expected columns and normalize dates/numeric series."""
    if "trade_date" not in df.columns:
        raise RuntimeError("Input is missing trade_date column")

    expected_columns = [
        *[f"ret_{sector}" for sector in STANDARD_SECTORS],
        *[f"sent_{sector}" for sector in STANDARD_SECTORS],
    ]
    missing_columns = [column for column in expected_columns if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Input is missing expected columns: {missing_columns}")

    prepared = df[["trade_date", *expected_columns]].copy()
    prepared["trade_date"] = pd.to_datetime(prepared["trade_date"], errors="coerce")
    invalid_dates = int(prepared["trade_date"].isna().sum())
    if invalid_dates:
        logger.warning("Dropping %d rows with invalid trade_date", invalid_dates)
    prepared = prepared.dropna(subset=["trade_date"]).copy()
    prepared["trade_date"] = prepared["trade_date"].dt.normalize()

    prepared = prepared.sort_values("trade_date").reset_index(drop=True)
    for column in expected_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")

    if fill_missing_sentiment:
        sent_columns = [f"sent_{sector}" for sector in STANDARD_SECTORS]
        filled = int(prepared[sent_columns].isna().sum().sum())
        prepared[sent_columns] = prepared[sent_columns].fillna(0.0)
        logger.info("Filled %d missing sentiment cells with 0 (neutral)", filled)
    return prepared


def _lagged_correlation(
    sent: pd.Series, ret: pd.Series, lag: int
) -> tuple[float, float, int]:
    """Pearson corr between sent(t) and ret(t+lag).

    Align sent[t] with ret shifted back by `lag` steps (ret.shift(-lag)),
    drop NaN, then run scipy pearsonr.
    """
    aligned = pd.concat(
        {"sent": sent, "ret": ret.shift(-lag)}, axis=1
    ).dropna()
    n_obs = len(aligned)
    if n_obs < MIN_OBS:
        return np.nan, np.nan, n_obs
    if aligned["sent"].std(ddof=0) <= 1e-12 or aligned["ret"].std(ddof=0) <= 1e-12:
        return np.nan, np.nan, n_obs
    corr, pvalue = pearsonr(aligned["sent"], aligned["ret"])
    return float(corr), float(pvalue), n_obs


def run_lead_lag(df: pd.DataFrame, max_lag: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute lead-lag cross-correlations for all (sentiment, return) sector pairs."""
    all_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    for s_sector in STANDARD_SECTORS:
        sent = df[f"sent_{s_sector}"]
        for r_sector in STANDARD_SECTORS:
            ret = df[f"ret_{r_sector}"]
            pair_rows: list[dict[str, Any]] = []
            for lag in range(0, max_lag + 1):
                corr, pvalue, n_obs = _lagged_correlation(sent, ret, lag)
                row = {
                    "s_sector": s_sector,
                    "r_sector": r_sector,
                    "lag": lag,
                    "corr": corr,
                    "pvalue": pvalue,
                    "significant": bool(pd.notna(pvalue) and pvalue < 0.05),
                    "n_obs": n_obs,
                }
                all_rows.append(row)
                pair_rows.append(row)

            valid = [r for r in pair_rows if pd.notna(r["corr"]) and r["n_obs"] >= MIN_OBS]
            if valid:
                best = max(valid, key=lambda item: abs(item["corr"]))
            else:
                # fall back to lag 0 placeholder so every pair has a best row
                best = pair_rows[0]
            best_rows.append(
                {
                    "s_sector": s_sector,
                    "r_sector": r_sector,
                    "best_lag": best["lag"],
                    "corr": best["corr"],
                    "pvalue": best["pvalue"],
                    "significant": best["significant"],
                    "n_obs": best["n_obs"],
                }
            )

    all_df = pd.DataFrame(all_rows)[ALL_COLUMNS]
    best_df = pd.DataFrame(best_rows)[BEST_COLUMNS]
    return all_df, best_df


def validate_outputs(all_df: pd.DataFrame, best_df: pd.DataFrame, max_lag: int) -> None:
    expected_all = 10 * 10 * (max_lag + 1)
    if len(all_df) != expected_all:
        raise RuntimeError(
            f"Expected {expected_all} lead_lag_all rows, found {len(all_df)}"
        )
    expected_best = 10 * 10
    if len(best_df) != expected_best:
        raise RuntimeError(
            f"Expected {expected_best} lead_lag_best rows, found {len(best_df)}"
        )
    corr_values = all_df["corr"].dropna()
    if not corr_values.empty and (corr_values.min() < -1.0 or corr_values.max() > 1.0):
        raise RuntimeError("corr values out of [-1, 1] range")


def print_summary(df: pd.DataFrame, all_df: pd.DataFrame, best_df: pd.DataFrame) -> None:
    print("input shape:", df.shape)
    print(
        "date range:",
        df["trade_date"].min().date(),
        "->",
        df["trade_date"].max().date(),
    )
    print("lead_lag_all output shape:", all_df.shape)
    print("lead_lag_best output shape:", best_df.shape)
    print(
        "number of significant relationships (all lags):",
        int(all_df["significant"].sum()),
    )

    leads = all_df[(all_df["lag"] >= 1) & all_df["significant"]].copy()
    leads = leads.dropna(subset=["corr"])
    leads["abs_corr"] = leads["corr"].abs()
    leads = leads.sort_values("abs_corr", ascending=False)
    print(
        "number of significant LEAD relationships (lag>=1):",
        int(len(leads)),
    )
    print("top 10 significant LEAD relationships (lag>=1, by |corr| desc):")
    if leads.empty:
        print("No significant lead relationships.")
    else:
        top = leads.head(10)[["s_sector", "r_sector", "lag", "corr", "pvalue", "n_obs"]]
        print(top.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--all-output", type=Path, default=DEFAULT_ALL_OUTPUT)
    parser.add_argument("--best-output", type=Path, default=DEFAULT_BEST_OUTPUT)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument(
        "--no-fill-missing",
        action="store_true",
        help="KHONG dien sentiment thieu = 0 (neutral)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_lag < 0:
            raise RuntimeError("--max-lag must be at least 0")

        raw_df = read_input(args.input)
        df = prepare_input(raw_df, fill_missing_sentiment=not args.no_fill_missing)
        all_df, best_df = run_lead_lag(df, args.max_lag)

        validate_outputs(all_df, best_df, args.max_lag)

        write_csv(all_df, args.all_output)
        write_csv(best_df, args.best_output)
        print_summary(df, all_df, best_df)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
