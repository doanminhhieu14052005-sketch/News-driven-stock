"""Build a price correlation network for the 10 HOSE sectors (RQ2 part 1)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/merged_sentiment_return_wide.csv")
DEFAULT_CORR_MATRIX_OUTPUT = Path("data/processed/price_correlation_matrix.csv")
DEFAULT_ADJACENCY_OUTPUT = Path("data/processed/price_network_adjacency.csv")
DEFAULT_EDGE_LIST_OUTPUT = Path("data/processed/price_network_edges.csv")
DEFAULT_NODE_METRICS_OUTPUT = Path("data/processed/price_network_node_metrics.csv")
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

EDGE_COLUMNS = ["source", "target", "corr", "abs_corr"]
NODE_METRIC_COLUMNS = ["sector", "degree", "weighted_degree", "avg_abs_corr"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module12_price_network")


def read_input(path: Path) -> pd.DataFrame:
    """Read merged sentiment-return data."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def prepare_returns(df: pd.DataFrame, dropna_rows: bool) -> pd.DataFrame:
    """Validate columns and return a numeric frame of the 10 ret_<sector> series."""
    if "trade_date" not in df.columns:
        raise RuntimeError("Input is missing trade_date column")

    ret_columns = [f"ret_{sector}" for sector in STANDARD_SECTORS]
    missing_columns = [column for column in ret_columns if column not in df.columns]
    if missing_columns:
        raise RuntimeError(f"Input is missing expected return columns: {missing_columns}")

    returns = df[ret_columns].copy()
    for column in ret_columns:
        returns[column] = pd.to_numeric(returns[column], errors="coerce")
    returns.columns = STANDARD_SECTORS

    if dropna_rows:
        before = len(returns)
        returns = returns.dropna(axis=0, how="any")
        dropped = before - len(returns)
        if dropped:
            logger.info("Dropped %d rows with missing returns (listwise)", dropped)

    if returns.empty:
        raise RuntimeError("No rows remain after removing missing returns")
    return returns


def compute_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix (pairwise) across the 10 sector return series."""
    corr = returns.corr(method="pearson", min_periods=1)
    corr = corr.reindex(index=STANDARD_SECTORS, columns=STANDARD_SECTORS)
    return corr


def _off_diagonal_abs(corr: pd.DataFrame) -> np.ndarray:
    """Absolute off-diagonal correlations as a flat array (each pair once)."""
    values = corr.to_numpy(dtype=float)
    iu = np.triu_indices(len(STANDARD_SECTORS), k=1)
    pair_values = np.abs(values[iu])
    return pair_values[~np.isnan(pair_values)]


def resolve_threshold(corr: pd.DataFrame, fixed_threshold: float | None) -> float:
    """Use a fixed threshold if provided, else the 75th percentile of |off-diagonal corr|."""
    if fixed_threshold is not None:
        return float(fixed_threshold)
    pair_values = _off_diagonal_abs(corr)
    if pair_values.size == 0:
        raise RuntimeError("No valid off-diagonal correlations to derive a threshold")
    return float(np.percentile(pair_values, 75))


def build_adjacency_matrix(corr: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Undirected binary adjacency: edge (i, j) = 1 if |corr_ij| > threshold."""
    abs_values = corr.abs().to_numpy(dtype=float)
    adjacency = (abs_values > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    adjacency[np.isnan(corr.to_numpy(dtype=float))] = 0
    return pd.DataFrame(adjacency, index=STANDARD_SECTORS, columns=STANDARD_SECTORS)


def build_edge_list(corr: pd.DataFrame, adjacency: pd.DataFrame) -> pd.DataFrame:
    """Edge list for adjacency == 1, each undirected pair once (upper triangle)."""
    rows: list[dict[str, Any]] = []
    for i, source in enumerate(STANDARD_SECTORS):
        for j in range(i + 1, len(STANDARD_SECTORS)):
            target = STANDARD_SECTORS[j]
            if int(adjacency.loc[source, target]) == 1:
                corr_value = float(corr.loc[source, target])
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "corr": corr_value,
                        "abs_corr": abs(corr_value),
                    }
                )
    if not rows:
        return pd.DataFrame(columns=EDGE_COLUMNS)
    edges = pd.DataFrame(rows)[EDGE_COLUMNS]
    return edges.sort_values("abs_corr", ascending=False).reset_index(drop=True)


def build_node_metrics(corr: pd.DataFrame, adjacency: pd.DataFrame) -> pd.DataFrame:
    """Compute degree, weighted_degree and avg_abs_corr per sector node."""
    abs_corr = corr.abs()
    rows: list[dict[str, Any]] = []
    for sector in STANDARD_SECTORS:
        neighbor_mask = adjacency.loc[sector] == 1
        degree = int(neighbor_mask.sum())
        weighted_degree = float(abs_corr.loc[sector][neighbor_mask].sum())
        off_diag = abs_corr.loc[sector].drop(labels=[sector])
        avg_abs_corr = float(off_diag.mean()) if off_diag.notna().any() else np.nan
        rows.append(
            {
                "sector": sector,
                "degree": degree,
                "weighted_degree": weighted_degree,
                "avg_abs_corr": avg_abs_corr,
            }
        )
    return pd.DataFrame(rows)[NODE_METRIC_COLUMNS]


def matrix_with_source_column(matrix: pd.DataFrame) -> pd.DataFrame:
    """Add the source_sector index column (matches module10 build_matrix layout)."""
    out = matrix.reset_index().rename(columns={"index": "source_sector"})
    return out[["source_sector", *STANDARD_SECTORS]]


def validate_outputs(
    corr: pd.DataFrame,
    adjacency: pd.DataFrame,
    edge_list: pd.DataFrame,
) -> None:
    if list(corr.index) != STANDARD_SECTORS or list(corr.columns) != STANDARD_SECTORS:
        raise RuntimeError("Correlation matrix is not a 10x10 sector matrix")
    if adjacency.shape != (10, 10):
        raise RuntimeError(f"Adjacency must be 10x10, found {adjacency.shape}")

    adj_values = set(np.unique(adjacency.to_numpy()).tolist())
    if not adj_values.issubset({0, 1}):
        raise RuntimeError(f"Adjacency contains non-binary values: {adj_values}")

    if not np.array_equal(adjacency.to_numpy(), adjacency.to_numpy().T):
        raise RuntimeError("Adjacency matrix is not symmetric (must be undirected)")

    diagonal = np.diag(adjacency.to_numpy())
    if diagonal.any():
        raise RuntimeError("Adjacency diagonal must be all zeros")

    invalid = sorted(
        (set(edge_list.get("source", pd.Series(dtype=str)))
         | set(edge_list.get("target", pd.Series(dtype=str))))
        - set(STANDARD_SECTORS)
    )
    if invalid:
        raise RuntimeError(f"Edge list contains invalid sectors: {invalid}")

    # Edge count must equal half the adjacency ones (undirected, no duplicates).
    expected_edges = int(adjacency.to_numpy().sum() // 2)
    if len(edge_list) != expected_edges:
        raise RuntimeError(
            f"Edge list has {len(edge_list)} edges, expected {expected_edges}"
        )


def print_summary(
    returns: pd.DataFrame,
    corr: pd.DataFrame,
    adjacency: pd.DataFrame,
    edge_list: pd.DataFrame,
    node_metrics: pd.DataFrame,
    threshold: float,
    threshold_source: str,
) -> None:
    print("return rows used:", len(returns))
    print("number of sectors:", len(STANDARD_SECTORS))
    print("correlation matrix shape:", corr.shape)
    print(f"threshold used: {threshold:.6f} ({threshold_source})")
    print("number of edges:", len(edge_list))
    possible = len(STANDARD_SECTORS) * (len(STANDARD_SECTORS) - 1) // 2
    print("possible undirected pairs:", possible)
    print("network density:", round(len(edge_list) / possible, 4) if possible else np.nan)
    print("top correlated sector pairs:")
    if edge_list.empty:
        print("No edges above threshold.")
    else:
        print(edge_list.head(10).to_string(index=False))
    print("node metrics:")
    print(node_metrics.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--corr-matrix-output", type=Path, default=DEFAULT_CORR_MATRIX_OUTPUT)
    parser.add_argument("--adjacency-output", type=Path, default=DEFAULT_ADJACENCY_OUTPUT)
    parser.add_argument("--edge-list-output", type=Path, default=DEFAULT_EDGE_LIST_OUTPUT)
    parser.add_argument("--node-metrics-output", type=Path, default=DEFAULT_NODE_METRICS_OUTPUT)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed |corr| edge threshold (e.g. 0.3). Default: 75th percentile of |off-diagonal corr|.",
    )
    parser.add_argument(
        "--pairwise",
        action="store_true",
        help="Keep pairwise correlations (do NOT drop rows listwise before correlating).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.threshold is not None and not (0.0 <= args.threshold <= 1.0):
            raise RuntimeError("--threshold must be between 0 and 1")

        raw_df = read_input(args.input)
        returns = prepare_returns(raw_df, dropna_rows=not args.pairwise)

        corr = compute_correlation_matrix(returns)
        threshold = resolve_threshold(corr, args.threshold)
        threshold_source = "fixed" if args.threshold is not None else "p75 |off-diag corr|"

        adjacency = build_adjacency_matrix(corr, threshold)
        edge_list = build_edge_list(corr, adjacency)
        node_metrics = build_node_metrics(corr, adjacency)

        validate_outputs(corr, adjacency, edge_list)

        write_csv(matrix_with_source_column(corr), args.corr_matrix_output)
        write_csv(matrix_with_source_column(adjacency), args.adjacency_output)
        write_csv(edge_list, args.edge_list_output)
        write_csv(node_metrics, args.node_metrics_output)

        print_summary(
            returns,
            corr,
            adjacency,
            edge_list,
            node_metrics,
            threshold,
            threshold_source,
        )
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
