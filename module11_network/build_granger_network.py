"""Build Granger spillover network tables and figures."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd


CSV_ENCODING = "utf-8-sig"
DEFAULT_GRANGER_RESULTS = Path("data/processed/granger_results.csv")
DEFAULT_PVALUE_MATRIX = Path("data/processed/granger_pvalue_matrix.csv")
DEFAULT_FDR_MATRIX = Path("data/processed/granger_fdr_matrix.csv")
DEFAULT_ADJACENCY_MATRIX = Path("data/processed/granger_adjacency_matrix.csv")
DEFAULT_OUTPUT_DIR = Path("data/processed")
DEFAULT_FIGURES_DIR = Path("reports/figures")

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

REQUIRED_GRANGER_COLUMNS = [
    "source_sector",
    "target_sector",
    "best_lag",
    "p_value",
    "p_value_fdr_bh",
    "f_stat",
    "n_obs",
    "significant_raw_05",
    "significant_fdr_05",
]

EDGE_COLUMNS = [
    "source",
    "target",
    "source_sector",
    "target_sector",
    "best_lag",
    "p_value",
    "p_value_fdr_bh",
    "f_stat",
    "n_obs",
    "edge_weight",
    "edge_type",
    "is_self_edge",
    "significance_type",
]

NODE_METRIC_COLUMNS = [
    "sector_label",
    "in_degree",
    "out_degree",
    "total_degree",
    "weighted_in_degree",
    "weighted_out_degree",
    "weighted_total_degree",
    "self_loop_count",
    "pagerank",
    "betweenness_centrality",
    "is_net_sender",
    "is_net_receiver",
]

SUMMARY_COLUMNS = [
    "network_type",
    "total_edges",
    "cross_edges",
    "self_edges",
    "number_of_nodes",
    "number_of_active_source_nodes",
    "number_of_active_target_nodes",
    "density_cross",
    "top_sender_by_weight",
    "top_receiver_by_weight",
    "strongest_edge_source",
    "strongest_edge_target",
    "strongest_edge_weight",
    "strongest_edge_p_value",
]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module11_network")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def coerce_bool(series: pd.Series) -> pd.Series:
    """Convert bool-like CSV values into real booleans."""
    if series.dtype == bool:
        return series.fillna(False)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.isin({"true", "1", "yes", "y"})


def validate_granger_results(results: pd.DataFrame) -> None:
    missing_columns = [column for column in REQUIRED_GRANGER_COLUMNS if column not in results.columns]
    if missing_columns:
        raise RuntimeError(f"granger_results.csv is missing columns: {missing_columns}")

    invalid_sectors = sorted(
        (
            set(results["source_sector"].dropna().astype(str))
            | set(results["target_sector"].dropna().astype(str))
        )
        - set(STANDARD_SECTORS)
    )
    if invalid_sectors:
        raise RuntimeError(f"granger_results.csv contains invalid sectors: {invalid_sectors}")


def prepare_granger_results(results: pd.DataFrame) -> pd.DataFrame:
    validate_granger_results(results)
    prepared = results.copy()
    for column in ["best_lag", "p_value", "p_value_fdr_bh", "f_stat", "n_obs"]:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
    prepared["significant_raw_05"] = coerce_bool(prepared["significant_raw_05"])
    prepared["significant_fdr_05"] = coerce_bool(prepared["significant_fdr_05"])
    return prepared


def compute_edge_weight(p_values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(p_values, errors="coerce")
    weights = pd.Series(np.nan, index=p_values.index, dtype=float)
    valid = numeric.notna()
    weights.loc[valid] = -np.log10(np.clip(numeric.loc[valid].astype(float), 1e-12, 1.0))
    return weights


def build_edge_table(
    results: pd.DataFrame,
    significance_column: str,
    pvalue_column: str,
    significance_type: str,
) -> pd.DataFrame:
    mask = coerce_bool(results[significance_column])
    edges = results.loc[mask].copy()
    if edges.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    edges["source"] = edges["source_sector"]
    edges["target"] = edges["target_sector"]
    edges["edge_weight"] = compute_edge_weight(edges[pvalue_column])
    edges["edge_type"] = np.where(edges["source_sector"].eq(edges["target_sector"]), "self", "cross")
    edges["is_self_edge"] = edges["edge_type"].eq("self")
    edges["significance_type"] = significance_type

    edges = edges[EDGE_COLUMNS].sort_values(
        ["edge_weight", "source_sector", "target_sector"],
        ascending=[False, True, True],
        na_position="last",
    )
    return edges.reset_index(drop=True)


def filter_plot_edges(
    edges: pd.DataFrame,
    include_self_loops: bool,
    min_edge_weight: float,
) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    plot_edges = edges.copy()
    if not include_self_loops:
        plot_edges = plot_edges.loc[~plot_edges["is_self_edge"]].copy()
    plot_edges = plot_edges.loc[
        plot_edges["edge_weight"].fillna(0).astype(float) >= min_edge_weight
    ].copy()
    return plot_edges


def build_cross_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(STANDARD_SECTORS)
    if edges.empty:
        return graph

    cross_edges = edges.loc[~edges["is_self_edge"]].copy()
    for _, row in cross_edges.iterrows():
        weight = row["edge_weight"]
        graph.add_edge(
            row["source_sector"],
            row["target_sector"],
            weight=float(weight) if pd.notna(weight) else 0.0,
            p_value=row["p_value"],
        )
    return graph


def build_node_metrics(edges: pd.DataFrame) -> pd.DataFrame:
    graph = build_cross_graph(edges)
    self_loop_counts = {sector: 0 for sector in STANDARD_SECTORS}
    if not edges.empty:
        self_edges = edges.loc[edges["is_self_edge"]]
        counts = self_edges["source_sector"].value_counts()
        self_loop_counts.update({sector: int(counts.get(sector, 0)) for sector in STANDARD_SECTORS})

    if graph.number_of_edges() > 0:
        try:
            pagerank = nx.pagerank(graph, weight="weight")
        except nx.NetworkXException as exc:
            logger.warning("PageRank failed, using zeros: %s", exc)
            pagerank = {sector: 0.0 for sector in STANDARD_SECTORS}
        betweenness = nx.betweenness_centrality(graph, normalized=True, weight=None)
    else:
        pagerank = {sector: 0.0 for sector in STANDARD_SECTORS}
        betweenness = {sector: 0.0 for sector in STANDARD_SECTORS}

    rows: list[dict[str, Any]] = []
    for sector in STANDARD_SECTORS:
        in_edges = list(graph.in_edges(sector, data=True))
        out_edges = list(graph.out_edges(sector, data=True))
        weighted_in = float(sum(data.get("weight", 0.0) for _, _, data in in_edges))
        weighted_out = float(sum(data.get("weight", 0.0) for _, _, data in out_edges))
        in_degree = len(in_edges)
        out_degree = len(out_edges)
        rows.append(
            {
                "sector_label": sector,
                "in_degree": in_degree,
                "out_degree": out_degree,
                "total_degree": in_degree + out_degree,
                "weighted_in_degree": weighted_in,
                "weighted_out_degree": weighted_out,
                "weighted_total_degree": weighted_in + weighted_out,
                "self_loop_count": self_loop_counts[sector],
                "pagerank": float(pagerank.get(sector, 0.0)),
                "betweenness_centrality": float(betweenness.get(sector, 0.0)),
                "is_net_sender": bool(weighted_out > weighted_in),
                "is_net_receiver": bool(weighted_in > weighted_out),
            }
        )
    return pd.DataFrame(rows)[NODE_METRIC_COLUMNS]


def build_summary_row(network_type: str, edges: pd.DataFrame, metrics: pd.DataFrame) -> dict[str, Any]:
    total_edges = int(len(edges))
    cross_edges = edges.loc[~edges["is_self_edge"]].copy() if not edges.empty else edges.copy()
    self_edges = edges.loc[edges["is_self_edge"]].copy() if not edges.empty else edges.copy()
    active_sources = set(cross_edges["source_sector"]) if not cross_edges.empty else set()
    active_targets = set(cross_edges["target_sector"]) if not cross_edges.empty else set()

    if not metrics.empty:
        sender_candidates = metrics.sort_values("weighted_out_degree", ascending=False)
        receiver_candidates = metrics.sort_values("weighted_in_degree", ascending=False)
        top_sender = sender_candidates.iloc[0]
        top_receiver = receiver_candidates.iloc[0]
        top_sender_label = top_sender["sector_label"] if top_sender["weighted_out_degree"] > 0 else ""
        top_receiver_label = top_receiver["sector_label"] if top_receiver["weighted_in_degree"] > 0 else ""
    else:
        top_sender_label = ""
        top_receiver_label = ""

    if edges.empty:
        strongest_source = ""
        strongest_target = ""
        strongest_weight = np.nan
        strongest_p_value = np.nan
    else:
        strongest = edges.sort_values("edge_weight", ascending=False, na_position="last").iloc[0]
        strongest_source = strongest["source_sector"]
        strongest_target = strongest["target_sector"]
        strongest_weight = strongest["edge_weight"]
        pvalue_column = "p_value_fdr_bh" if network_type == "fdr" else "p_value"
        strongest_p_value = strongest[pvalue_column]

    return {
        "network_type": network_type,
        "total_edges": total_edges,
        "cross_edges": int(len(cross_edges)),
        "self_edges": int(len(self_edges)),
        "number_of_nodes": len(STANDARD_SECTORS),
        "number_of_active_source_nodes": len(active_sources),
        "number_of_active_target_nodes": len(active_targets),
        "density_cross": len(cross_edges) / (len(STANDARD_SECTORS) * (len(STANDARD_SECTORS) - 1)),
        "top_sender_by_weight": top_sender_label,
        "top_receiver_by_weight": top_receiver_label,
        "strongest_edge_source": strongest_source,
        "strongest_edge_target": strongest_target,
        "strongest_edge_weight": strongest_weight,
        "strongest_edge_p_value": strongest_p_value,
    }


def build_network_summary(
    raw_edges: pd.DataFrame,
    fdr_edges: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    fdr_metrics: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        build_summary_row("raw", raw_edges, raw_metrics),
        build_summary_row("fdr", fdr_edges, fdr_metrics),
    ]
    return pd.DataFrame(rows)[SUMMARY_COLUMNS]


def build_adjacency_from_results(results: pd.DataFrame, significance_column: str) -> pd.DataFrame:
    matrix = pd.DataFrame(0, index=STANDARD_SECTORS, columns=STANDARD_SECTORS, dtype=int)
    mask = coerce_bool(results[significance_column])
    for _, row in results.loc[mask].iterrows():
        source = row["source_sector"]
        target = row["target_sector"]
        if source in STANDARD_SECTORS and target in STANDARD_SECTORS:
            matrix.loc[source, target] = 1
    return matrix.reset_index().rename(columns={"index": "source_sector"})


def normalize_matrix(matrix: pd.DataFrame, name: str) -> pd.DataFrame:
    if "source_sector" not in matrix.columns:
        raise RuntimeError(f"{name} matrix is missing source_sector column")
    missing_columns = [sector for sector in STANDARD_SECTORS if sector not in matrix.columns]
    if missing_columns:
        raise RuntimeError(f"{name} matrix is missing sector columns: {missing_columns}")

    normalized = matrix[["source_sector", *STANDARD_SECTORS]].copy()
    normalized["source_sector"] = normalized["source_sector"].astype(str)
    missing_rows = [sector for sector in STANDARD_SECTORS if sector not in set(normalized["source_sector"])]
    if missing_rows:
        raise RuntimeError(f"{name} matrix is missing source rows: {missing_rows}")
    normalized = normalized.set_index("source_sector").loc[STANDARD_SECTORS].reset_index()
    for sector in STANDARD_SECTORS:
        normalized[sector] = pd.to_numeric(normalized[sector], errors="coerce")
    return normalized


def matrix_values(matrix: pd.DataFrame) -> np.ndarray:
    return matrix[STANDARD_SECTORS].to_numpy(dtype=float)


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: Path,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = matrix_values(matrix)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(STANDARD_SECTORS)))
    ax.set_yticks(range(len(STANDARD_SECTORS)))
    ax.set_xticklabels(STANDARD_SECTORS, rotation=45, ha="right")
    ax.set_yticklabels(STANDARD_SECTORS)
    ax.set_xlabel("Target sector return")
    ax.set_ylabel("Source sector sentiment")
    ax.set_title(title)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_network(
    edges: pd.DataFrame,
    metrics: pd.DataFrame,
    output_path: Path,
    title: str,
    empty_message: str,
    include_self_loops: bool,
    min_edge_weight: float,
    top_n_labels: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_edges = filter_plot_edges(edges, include_self_loops, min_edge_weight)
    graph = nx.DiGraph()
    graph.add_nodes_from(STANDARD_SECTORS)
    for _, row in plot_edges.iterrows():
        graph.add_edge(
            row["source_sector"],
            row["target_sector"],
            weight=float(row["edge_weight"]) if pd.notna(row["edge_weight"]) else 0.0,
            best_lag=row["best_lag"],
        )

    node_weight = dict(zip(metrics["sector_label"], metrics["weighted_total_degree"]))
    max_node_weight = max(node_weight.values()) if node_weight else 0.0
    node_sizes = []
    for sector in STANDARD_SECTORS:
        if max_node_weight > 0:
            node_sizes.append(1200 + 2200 * (node_weight.get(sector, 0.0) / max_node_weight))
        else:
            node_sizes.append(1600)

    pos = nx.circular_layout(graph)
    fig, ax = plt.subplots(figsize=(12, 10))
    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=node_sizes,
        node_color="#dbeafe",
        edgecolors="#1f2937",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(graph, pos, font_size=9, font_weight="bold", ax=ax)

    if graph.number_of_edges() > 0:
        weights = [data.get("weight", 0.0) for _, _, data in graph.edges(data=True)]
        max_weight = max(weights) if weights else 0.0
        widths = [1.0 + 5.0 * (weight / max_weight) if max_weight > 0 else 1.0 for weight in weights]
        nx.draw_networkx_edges(
            graph,
            pos,
            width=widths,
            edge_color="#2563eb",
            alpha=0.72,
            arrows=True,
            arrowsize=18,
            arrowstyle="-|>",
            connectionstyle="arc3,rad=0.08",
            ax=ax,
        )

        if top_n_labels > 0:
            top_edges = plot_edges.sort_values("edge_weight", ascending=False).head(top_n_labels)
            edge_labels = {
                (row["source_sector"], row["target_sector"]): f"lag {int(row['best_lag'])}"
                for _, row in top_edges.iterrows()
                if pd.notna(row["best_lag"])
            }
            nx.draw_networkx_edge_labels(
                graph,
                pos,
                edge_labels=edge_labels,
                font_size=7,
                label_pos=0.58,
                ax=ax,
            )
    else:
        ax.text(
            0.5,
            0.5,
            empty_message,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=15,
            color="#991b1b",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fee2e2", "edgecolor": "#991b1b"},
        )

    plot_title = title
    excluded_self_loops = int(edges["is_self_edge"].sum()) if not edges.empty and not include_self_loops else 0
    if excluded_self_loops:
        plot_title += "\nSelf-loops retained in CSV, excluded from plot."
    ax.set_title(plot_title, fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def validate_outputs(
    results: pd.DataFrame,
    raw_edges: pd.DataFrame,
    fdr_edges: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    fdr_metrics: pd.DataFrame,
    raw_adjacency: pd.DataFrame,
    fdr_adjacency: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    raw_expected = int(coerce_bool(results["significant_raw_05"]).sum())
    fdr_expected = int(coerce_bool(results["significant_fdr_05"]).sum())
    if len(raw_edges) != raw_expected:
        raise RuntimeError(f"Expected {raw_expected} raw edges, found {len(raw_edges)}")
    if len(fdr_edges) != fdr_expected:
        raise RuntimeError(f"Expected {fdr_expected} FDR edges, found {len(fdr_edges)}")
    if len(raw_metrics) != len(STANDARD_SECTORS):
        raise RuntimeError("Raw node metrics must contain 10 sector rows")
    if len(fdr_metrics) != len(STANDARD_SECTORS):
        raise RuntimeError("FDR node metrics must contain 10 sector rows")

    normalize_matrix(raw_adjacency, "raw adjacency")
    normalize_matrix(fdr_adjacency, "fdr adjacency")

    for path in figure_paths:
        if not path.exists() or path.stat().st_size <= 0:
            raise RuntimeError(f"Figure was not created or is empty: {path.as_posix()}")


def print_top_edges(label: str, edges: pd.DataFrame) -> None:
    print(label)
    if edges.empty:
        print("No edges.")
    else:
        print(
            edges[
                [
                    "source_sector",
                    "target_sector",
                    "best_lag",
                    "p_value",
                    "p_value_fdr_bh",
                    "edge_weight",
                    "edge_type",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )


def print_top_nodes(label: str, metrics: pd.DataFrame, column: str) -> None:
    print(label)
    ordered = metrics.sort_values(column, ascending=False)
    active = ordered.loc[ordered[column] > 0]
    if active.empty:
        print("No active nodes.")
    else:
        print(active[["sector_label", column]].head(10).to_string(index=False))


def print_summary(
    results: pd.DataFrame,
    raw_edges: pd.DataFrame,
    fdr_edges: pd.DataFrame,
    raw_metrics: pd.DataFrame,
    fdr_metrics: pd.DataFrame,
    output_paths: list[Path],
    figure_paths: list[Path],
) -> None:
    print("granger_results shape:", results.shape)
    print("raw edges count:", len(raw_edges))
    print("raw cross edges count:", int((~raw_edges["is_self_edge"]).sum()) if not raw_edges.empty else 0)
    print("raw self edges count:", int(raw_edges["is_self_edge"].sum()) if not raw_edges.empty else 0)
    print("fdr edges count:", len(fdr_edges))
    print("fdr cross edges count:", int((~fdr_edges["is_self_edge"]).sum()) if not fdr_edges.empty else 0)
    print("fdr self edges count:", int(fdr_edges["is_self_edge"].sum()) if not fdr_edges.empty else 0)
    print_top_edges("top raw edges by edge_weight:", raw_edges)
    print_top_nodes("top raw senders:", raw_metrics, "weighted_out_degree")
    print_top_nodes("top raw receivers:", raw_metrics, "weighted_in_degree")
    print_top_nodes("top fdr senders:", fdr_metrics, "weighted_out_degree")
    print("output file paths:")
    for path in output_paths:
        print(path.as_posix())
    print("figure paths:")
    for path in figure_paths:
        print(path.as_posix())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granger-results", type=Path, default=DEFAULT_GRANGER_RESULTS)
    parser.add_argument("--pvalue-matrix", type=Path, default=DEFAULT_PVALUE_MATRIX)
    parser.add_argument("--fdr-matrix", type=Path, default=DEFAULT_FDR_MATRIX)
    parser.add_argument("--adjacency-matrix", type=Path, default=DEFAULT_ADJACENCY_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_FIGURES_DIR)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--include-self-loops-in-plot", action="store_true")
    parser.add_argument("--top-n-labels", type=int, default=10)
    parser.add_argument("--min-edge-weight", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not (0 < args.alpha < 1):
            raise RuntimeError("--alpha must be between 0 and 1")
        if args.top_n_labels < 0:
            raise RuntimeError("--top-n-labels must be non-negative")
        if args.min_edge_weight < 0:
            raise RuntimeError("--min-edge-weight must be non-negative")

        args.output_dir.mkdir(parents=True, exist_ok=True)
        args.figures_dir.mkdir(parents=True, exist_ok=True)

        results = prepare_granger_results(read_csv(args.granger_results))
        pvalue_matrix = normalize_matrix(read_csv(args.pvalue_matrix), "raw p-value")
        fdr_matrix = normalize_matrix(read_csv(args.fdr_matrix), "fdr p-value")
        fdr_adjacency_from_file = normalize_matrix(read_csv(args.adjacency_matrix), "fdr adjacency")

        raw_edges = build_edge_table(results, "significant_raw_05", "p_value", "raw")
        fdr_edges = build_edge_table(results, "significant_fdr_05", "p_value_fdr_bh", "fdr")
        raw_metrics = build_node_metrics(raw_edges)
        fdr_metrics = build_node_metrics(fdr_edges)
        summary = build_network_summary(raw_edges, fdr_edges, raw_metrics, fdr_metrics)
        raw_adjacency = build_adjacency_from_results(results, "significant_raw_05")
        fdr_adjacency = build_adjacency_from_results(results, "significant_fdr_05")

        output_paths = [
            args.output_dir / "network_edges_raw.csv",
            args.output_dir / "network_edges_fdr.csv",
            args.output_dir / "network_node_metrics_raw.csv",
            args.output_dir / "network_node_metrics_fdr.csv",
            args.output_dir / "network_summary.csv",
        ]
        figure_paths = [
            args.figures_dir / "granger_network_raw.png",
            args.figures_dir / "granger_network_fdr.png",
            args.figures_dir / "granger_pvalue_heatmap.png",
            args.figures_dir / "granger_fdr_heatmap.png",
            args.figures_dir / "granger_raw_adjacency_heatmap.png",
            args.figures_dir / "granger_fdr_adjacency_heatmap.png",
        ]

        write_csv(raw_edges, output_paths[0])
        write_csv(fdr_edges, output_paths[1])
        write_csv(raw_metrics, output_paths[2])
        write_csv(fdr_metrics, output_paths[3])
        write_csv(summary, output_paths[4])

        plot_network(
            raw_edges,
            raw_metrics,
            figure_paths[0],
            "Raw exploratory Granger network, uncorrected p < 0.05",
            "No raw significant edges",
            args.include_self_loops_in_plot,
            args.min_edge_weight,
            args.top_n_labels,
        )
        plot_network(
            fdr_edges,
            fdr_metrics,
            figure_paths[1],
            "FDR-confirmed Granger network, BH-adjusted p < 0.05",
            "No FDR-significant edges",
            args.include_self_loops_in_plot,
            args.min_edge_weight,
            args.top_n_labels,
        )
        plot_heatmap(
            pvalue_matrix,
            figure_paths[2],
            "Raw Granger p-value matrix",
            "Raw p-value",
            "viridis_r",
            vmin=0,
            vmax=1,
        )
        plot_heatmap(
            fdr_matrix,
            figure_paths[3],
            "FDR-adjusted Granger p-value matrix",
            "BH-adjusted p-value",
            "viridis_r",
            vmin=0,
            vmax=1,
        )
        plot_heatmap(
            raw_adjacency,
            figure_paths[4],
            "Raw Granger adjacency matrix",
            "Significant raw edge",
            "Blues",
            vmin=0,
            vmax=1,
        )
        plot_heatmap(
            fdr_adjacency_from_file,
            figure_paths[5],
            "FDR Granger adjacency matrix",
            "Significant FDR edge",
            "Blues",
            vmin=0,
            vmax=1,
        )

        validate_outputs(
            results,
            raw_edges,
            fdr_edges,
            raw_metrics,
            fdr_metrics,
            raw_adjacency,
            fdr_adjacency,
            figure_paths,
        )
        print_summary(results, raw_edges, fdr_edges, raw_metrics, fdr_metrics, output_paths, figure_paths)
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
