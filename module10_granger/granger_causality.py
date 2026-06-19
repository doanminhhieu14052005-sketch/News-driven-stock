"""Run stationarity diagnostics and Granger causality tests."""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests
from statsmodels.tsa.stattools import adfuller, grangercausalitytests


DEFAULT_INPUT = Path("data/processed/merged_sentiment_return_wide.csv")
DEFAULT_STATIONARITY_OUTPUT = Path("data/processed/stationarity_tests.csv")
DEFAULT_ALL_LAGS_OUTPUT = Path("data/processed/granger_all_lags.csv")
DEFAULT_RESULTS_OUTPUT = Path("data/processed/granger_results.csv")
DEFAULT_PVALUE_MATRIX_OUTPUT = Path("data/processed/granger_pvalue_matrix.csv")
DEFAULT_FDR_MATRIX_OUTPUT = Path("data/processed/granger_fdr_matrix.csv")
DEFAULT_ADJACENCY_OUTPUT = Path("data/processed/granger_adjacency_matrix.csv")
DEFAULT_EDGE_LIST_OUTPUT = Path("data/processed/granger_edge_list.csv")
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

STATIONARITY_COLUMNS = [
    "variable",
    "variable_type",
    "sector",
    "n_obs",
    "missing_count",
    "missing_ratio",
    "adf_statistic",
    "adf_pvalue",
    "adf_used_lag",
    "adf_nobs",
    "is_stationary_5pct",
    "status",
]

ALL_LAGS_COLUMNS = [
    "source_sector",
    "target_sector",
    "source_variable",
    "target_variable",
    "lag",
    "n_obs",
    "f_stat",
    "p_value",
    "df_denom",
    "df_num",
    "status",
    "error_message",
]

RESULT_COLUMNS = [
    "source_sector",
    "target_sector",
    "source_variable",
    "target_variable",
    "best_lag",
    "n_obs",
    "f_stat",
    "p_value",
    "p_value_fdr_bh",
    "significant_raw_05",
    "significant_fdr_05",
    "alpha",
    "lag_selection_method",
    "status",
    "error_message",
    "edge_weight",
]

EDGE_COLUMNS = [
    "source",
    "target",
    "source_sector",
    "target_sector",
    "best_lag",
    "p_value",
    "p_value_fdr_bh",
    "edge_weight",
    "n_obs",
    "f_stat",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module10_granger")


def read_input(path: Path) -> pd.DataFrame:
    """Read merged sentiment-return data."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def _parse_bound(value: str | None, name: str) -> pd.Timestamp | None:
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise RuntimeError(f"Invalid {name}: {value}")
    return pd.Timestamp(parsed).normalize()


def prepare_input(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
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

    lower = _parse_bound(start_date, "start-date")
    upper = _parse_bound(end_date, "end-date")
    if lower is not None:
        prepared = prepared.loc[prepared["trade_date"] >= lower].copy()
    if upper is not None:
        prepared = prepared.loc[prepared["trade_date"] <= upper].copy()
    if prepared.empty:
        raise RuntimeError("No rows remain after date filtering")

    prepared = prepared.sort_values("trade_date").reset_index(drop=True)
    for column in expected_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    return prepared


def _is_constant(series: pd.Series) -> bool:
    clean = series.dropna()
    if clean.nunique() <= 1:
        return True
    return bool(clean.std(ddof=0) <= 1e-12)


def run_stationarity_tests(df: pd.DataFrame, min_observations: int) -> pd.DataFrame:
    """Run ADF tests for all return and sentiment variables."""
    rows: list[dict[str, Any]] = []
    total_rows = len(df)
    for variable_type, prefix in [("return", "ret"), ("sentiment", "sent")]:
        for sector in STANDARD_SECTORS:
            variable = f"{prefix}_{sector}"
            series = df[variable].dropna()
            n_obs = len(series)
            missing_count = total_rows - n_obs
            base = {
                "variable": variable,
                "variable_type": variable_type,
                "sector": sector,
                "n_obs": n_obs,
                "missing_count": missing_count,
                "missing_ratio": missing_count / total_rows if total_rows else np.nan,
                "adf_statistic": np.nan,
                "adf_pvalue": np.nan,
                "adf_used_lag": np.nan,
                "adf_nobs": np.nan,
                "is_stationary_5pct": False,
                "status": "ok",
            }
            if n_obs < min_observations:
                base["status"] = "insufficient_observations"
                rows.append(base)
                continue
            if _is_constant(series):
                base["status"] = "constant_series"
                rows.append(base)
                continue
            try:
                stat, pvalue, used_lag, adf_nobs, *_ = adfuller(series, autolag="AIC")
                base.update(
                    {
                        "adf_statistic": stat,
                        "adf_pvalue": pvalue,
                        "adf_used_lag": used_lag,
                        "adf_nobs": adf_nobs,
                        "is_stationary_5pct": bool(pvalue < 0.05),
                    }
                )
            except Exception as exc:  # statsmodels can raise numerical edge cases
                base["status"] = "error"
                logger.warning("ADF failed for %s: %s", variable, exc)
            rows.append(base)
    return pd.DataFrame(rows)[STATIONARITY_COLUMNS]


def _empty_all_lags_row(
    source_sector: str,
    target_sector: str,
    source_variable: str,
    target_variable: str,
    n_obs: int,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "source_sector": source_sector,
        "target_sector": target_sector,
        "source_variable": source_variable,
        "target_variable": target_variable,
        "lag": np.nan,
        "n_obs": n_obs,
        "f_stat": np.nan,
        "p_value": np.nan,
        "df_denom": np.nan,
        "df_num": np.nan,
        "status": status,
        "error_message": error_message,
    }


def _empty_result_row(
    source_sector: str,
    target_sector: str,
    source_variable: str,
    target_variable: str,
    n_obs: int,
    alpha: float,
    status: str,
    error_message: str = "",
) -> dict[str, Any]:
    return {
        "source_sector": source_sector,
        "target_sector": target_sector,
        "source_variable": source_variable,
        "target_variable": target_variable,
        "best_lag": np.nan,
        "n_obs": n_obs,
        "f_stat": np.nan,
        "p_value": np.nan,
        "p_value_fdr_bh": np.nan,
        "significant_raw_05": False,
        "significant_fdr_05": False,
        "alpha": alpha,
        "lag_selection_method": "min_pvalue",
        "status": status,
        "error_message": error_message,
        "edge_weight": np.nan,
    }


def _effective_max_lag(n_obs: int, requested_max_lag: int) -> int:
    """Choose a conservative max lag to avoid overfitting/numerical failures."""
    safe_lag = max(0, (n_obs - 3) // 3)
    return int(min(requested_max_lag, safe_lag))


def run_granger_tests(
    df: pd.DataFrame,
    max_lag: int,
    min_observations: int,
    alpha: float,
    exclude_self: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run sent_source -> ret_target Granger tests for all sector pairs."""
    all_lag_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []

    for source_sector in STANDARD_SECTORS:
        source_variable = f"sent_{source_sector}"
        for target_sector in STANDARD_SECTORS:
            if exclude_self and source_sector == target_sector:
                continue
            target_variable = f"ret_{target_sector}"
            pair_df = df[[target_variable, source_variable]].dropna()
            n_obs = len(pair_df)

            if n_obs < min_observations:
                status = "insufficient_observations"
                all_lag_rows.append(
                    _empty_all_lags_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        status,
                    )
                )
                result_rows.append(
                    _empty_result_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        alpha,
                        status,
                    )
                )
                continue

            if _is_constant(pair_df[target_variable]) or _is_constant(pair_df[source_variable]):
                status = "constant_series"
                all_lag_rows.append(
                    _empty_all_lags_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        status,
                    )
                )
                result_rows.append(
                    _empty_result_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        alpha,
                        status,
                    )
                )
                continue

            effective_lag = _effective_max_lag(n_obs, max_lag)
            if effective_lag < 1:
                status = "insufficient_observations"
                all_lag_rows.append(
                    _empty_all_lags_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        status,
                        "effective_max_lag < 1",
                    )
                )
                result_rows.append(
                    _empty_result_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        alpha,
                        status,
                        "effective_max_lag < 1",
                    )
                )
                continue

            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message="verbose is deprecated since functions should not print results",
                        category=FutureWarning,
                    )
                    test_results = grangercausalitytests(
                        pair_df[[target_variable, source_variable]],
                        maxlag=list(range(1, effective_lag + 1)),
                        verbose=False,
                    )
                valid_lags: list[dict[str, Any]] = []
                for lag in range(1, effective_lag + 1):
                    ssr_ftest = test_results[lag][0]["ssr_ftest"]
                    f_stat, p_value, df_denom, df_num = ssr_ftest
                    row = {
                        "source_sector": source_sector,
                        "target_sector": target_sector,
                        "source_variable": source_variable,
                        "target_variable": target_variable,
                        "lag": lag,
                        "n_obs": n_obs,
                        "f_stat": f_stat,
                        "p_value": p_value,
                        "df_denom": df_denom,
                        "df_num": df_num,
                        "status": "ok",
                        "error_message": "",
                    }
                    all_lag_rows.append(row)
                    valid_lags.append(row)

                best = min(valid_lags, key=lambda item: item["p_value"])
                result_rows.append(
                    {
                        "source_sector": source_sector,
                        "target_sector": target_sector,
                        "source_variable": source_variable,
                        "target_variable": target_variable,
                        "best_lag": best["lag"],
                        "n_obs": n_obs,
                        "f_stat": best["f_stat"],
                        "p_value": best["p_value"],
                        "p_value_fdr_bh": np.nan,
                        "significant_raw_05": bool(best["p_value"] < alpha),
                        "significant_fdr_05": False,
                        "alpha": alpha,
                        "lag_selection_method": "min_pvalue",
                        "status": "ok",
                        "error_message": "",
                        "edge_weight": np.nan,
                    }
                )
            except Exception as exc:
                message = str(exc)
                logger.warning(
                    "Granger failed for %s -> %s: %s",
                    source_variable,
                    target_variable,
                    message,
                )
                all_lag_rows.append(
                    _empty_all_lags_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        "error",
                        message,
                    )
                )
                result_rows.append(
                    _empty_result_row(
                        source_sector,
                        target_sector,
                        source_variable,
                        target_variable,
                        n_obs,
                        alpha,
                        "error",
                        message,
                    )
                )

    all_lags = pd.DataFrame(all_lag_rows)[ALL_LAGS_COLUMNS]
    results = pd.DataFrame(result_rows)[RESULT_COLUMNS]
    results = apply_fdr_correction(results, alpha)
    return all_lags, results


def apply_fdr_correction(results: pd.DataFrame, alpha: float) -> pd.DataFrame:
    """Apply Benjamini-Hochberg FDR correction across successful pairs."""
    corrected = results.copy()
    ok_mask = corrected["status"].eq("ok") & corrected["p_value"].notna()
    if ok_mask.any():
        rejected, pvals_corrected, _, _ = multipletests(
            corrected.loc[ok_mask, "p_value"].astype(float),
            alpha=alpha,
            method="fdr_bh",
        )
        corrected.loc[ok_mask, "p_value_fdr_bh"] = pvals_corrected
        corrected.loc[ok_mask, "significant_fdr_05"] = rejected
        clipped = np.clip(pvals_corrected, 1e-12, 1.0)
        corrected.loc[ok_mask, "edge_weight"] = -np.log10(clipped)
    corrected["significant_raw_05"] = corrected["p_value"].lt(alpha).fillna(False)
    corrected["significant_fdr_05"] = corrected["significant_fdr_05"].fillna(False)
    return corrected[RESULT_COLUMNS]


def build_matrix(results: pd.DataFrame, value_column: str, default_value: Any = np.nan) -> pd.DataFrame:
    """Build a 10x10 source-target matrix from pairwise results."""
    matrix = pd.DataFrame(default_value, index=STANDARD_SECTORS, columns=STANDARD_SECTORS)
    for _, row in results.iterrows():
        source = row["source_sector"]
        target = row["target_sector"]
        if source in STANDARD_SECTORS and target in STANDARD_SECTORS:
            matrix.loc[source, target] = row[value_column]
    matrix = matrix.reset_index().rename(columns={"index": "source_sector"})
    return matrix[["source_sector", *STANDARD_SECTORS]]


def build_adjacency_matrix(results: pd.DataFrame) -> pd.DataFrame:
    matrix = build_matrix(results, "significant_fdr_05", default_value=False)
    for sector in STANDARD_SECTORS:
        matrix[sector] = matrix[sector].fillna(False).astype(bool).astype(int)
    return matrix


def build_edge_list(results: pd.DataFrame) -> pd.DataFrame:
    """Create significant FDR edge list for Module 11."""
    edges = results.loc[results["significant_fdr_05"].fillna(False)].copy()
    if edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    edges["source"] = edges["source_sector"]
    edges["target"] = edges["target_sector"]
    edges = edges.sort_values("edge_weight", ascending=False)
    return edges[EDGE_COLUMNS]


def validate_outputs(
    results: pd.DataFrame,
    pvalue_matrix: pd.DataFrame,
    fdr_matrix: pd.DataFrame,
    adjacency_matrix: pd.DataFrame,
    edge_list: pd.DataFrame,
    exclude_self: bool,
) -> None:
    expected_pairs = 90 if exclude_self else 100
    if len(results) != expected_pairs:
        raise RuntimeError(f"Expected {expected_pairs} Granger result rows, found {len(results)}")
    for name, matrix in {
        "pvalue": pvalue_matrix,
        "fdr": fdr_matrix,
        "adjacency": adjacency_matrix,
    }.items():
        expected_columns = ["source_sector", *STANDARD_SECTORS]
        missing_columns = [column for column in expected_columns if column not in matrix.columns]
        if missing_columns:
            raise RuntimeError(f"{name} matrix is missing columns: {missing_columns}")
        if len(matrix) != 10:
            raise RuntimeError(f"{name} matrix must have 10 source rows")
    adjacency_values = set(adjacency_matrix[STANDARD_SECTORS].stack().dropna().unique().tolist())
    if not adjacency_values.issubset({0, 1}):
        raise RuntimeError(f"Adjacency matrix contains non-binary values: {adjacency_values}")
    invalid_edges = sorted(
        (set(edge_list.get("source_sector", pd.Series(dtype=str))) | set(edge_list.get("target_sector", pd.Series(dtype=str))))
        - set(STANDARD_SECTORS)
    )
    if invalid_edges:
        raise RuntimeError(f"Edge list contains invalid sectors: {invalid_edges}")


def print_summary(
    df: pd.DataFrame,
    stationarity: pd.DataFrame,
    all_lags: pd.DataFrame,
    results: pd.DataFrame,
    pvalue_matrix: pd.DataFrame,
    adjacency_matrix: pd.DataFrame,
    edge_list: pd.DataFrame,
) -> None:
    print("input shape:", df.shape)
    print("date range:", df["trade_date"].min().date(), "->", df["trade_date"].max().date())
    print("number of return columns:", len([c for c in df.columns if c.startswith("ret_")]))
    print("number of sentiment columns:", len([c for c in df.columns if c.startswith("sent_")]))
    print("stationarity output shape:", stationarity.shape)
    print("all lags output shape:", all_lags.shape)
    print("results output shape:", results.shape)
    print("pvalue matrix shape:", pvalue_matrix.shape)
    print("adjacency matrix shape:", adjacency_matrix.shape)
    print("number of valid Granger pairs:", int(results["status"].eq("ok").sum()))
    print("number of significant raw pairs:", int(results["significant_raw_05"].sum()))
    print("number of significant FDR pairs:", int(results["significant_fdr_05"].sum()))
    print("top 10 strongest edges by edge_weight:")
    if edge_list.empty:
        print("No significant FDR edges.")
    else:
        print(edge_list.head(10).to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--stationarity-output", type=Path, default=DEFAULT_STATIONARITY_OUTPUT)
    parser.add_argument("--all-lags-output", type=Path, default=DEFAULT_ALL_LAGS_OUTPUT)
    parser.add_argument("--results-output", type=Path, default=DEFAULT_RESULTS_OUTPUT)
    parser.add_argument("--pvalue-matrix-output", type=Path, default=DEFAULT_PVALUE_MATRIX_OUTPUT)
    parser.add_argument("--fdr-matrix-output", type=Path, default=DEFAULT_FDR_MATRIX_OUTPUT)
    parser.add_argument("--adjacency-output", type=Path, default=DEFAULT_ADJACENCY_OUTPUT)
    parser.add_argument("--edge-list-output", type=Path, default=DEFAULT_EDGE_LIST_OUTPUT)
    parser.add_argument("--max-lag", type=int, default=5)
    parser.add_argument("--min-observations", type=int, default=60)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--exclude-self", action="store_true")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.max_lag < 1:
            raise RuntimeError("--max-lag must be at least 1")
        if args.min_observations < 10:
            raise RuntimeError("--min-observations must be at least 10")
        if not (0 < args.alpha < 1):
            raise RuntimeError("--alpha must be between 0 and 1")

        raw_df = read_input(args.input)
        df = prepare_input(raw_df, args.start_date, args.end_date)
        stationarity = run_stationarity_tests(df, args.min_observations)
        all_lags, results = run_granger_tests(
            df,
            args.max_lag,
            args.min_observations,
            args.alpha,
            args.exclude_self,
        )
        pvalue_matrix = build_matrix(results, "p_value")
        fdr_matrix = build_matrix(results, "p_value_fdr_bh")
        adjacency_matrix = build_adjacency_matrix(results)
        edge_list = build_edge_list(results)

        validate_outputs(
            results,
            pvalue_matrix,
            fdr_matrix,
            adjacency_matrix,
            edge_list,
            args.exclude_self,
        )

        write_csv(stationarity, args.stationarity_output)
        write_csv(all_lags, args.all_lags_output)
        write_csv(results, args.results_output)
        write_csv(pvalue_matrix, args.pvalue_matrix_output)
        write_csv(fdr_matrix, args.fdr_matrix_output)
        write_csv(adjacency_matrix, args.adjacency_output)
        write_csv(edge_list, args.edge_list_output)
        print_summary(
            df,
            stationarity,
            all_lags,
            results,
            pvalue_matrix,
            adjacency_matrix,
            edge_list,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
