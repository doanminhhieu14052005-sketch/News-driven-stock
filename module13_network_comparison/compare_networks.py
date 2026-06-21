"""So sánh mạng tin tức (sentiment spillover) với mạng giá (price correlation) — RQ2.

So 2 ma trận kề (adjacency) nhị phân của cùng 10 ngành bằng:
  - Frobenius distance / similarity
  - Jaccard similarity (độ trùng cạnh)
  - Tương quan Pearson & Spearman của các ô off-diagonal
  - QAP / Mantel permutation test (hoán vị nhãn node để tạo phân phối null)

Mạng tin tức (có hướng từ Granger) được đối xứng hóa (cạnh i-j nếu i->j HOẶC j->i)
để so với mạng giá (vô hướng).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

DEFAULT_NEWS_RESULTS = Path("data/processed/granger_results.csv")
DEFAULT_PRICE_ADJACENCY = Path("data/processed/price_network_adjacency.csv")
DEFAULT_OUTPUT = Path("data/processed/network_comparison.csv")
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

N_PERMUTATIONS = 2000
RANDOM_SEED = 42

try:  # in tiếng Việt không lỗi trên Windows (cp1252)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module13_network_comparison")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def build_news_adjacency(results: pd.DataFrame, significance_column: str) -> np.ndarray:
    """Ma trận kề CÓ HƯỚNG của mạng tin tức từ kết quả Granger."""
    if significance_column not in results.columns:
        raise RuntimeError(f"news results thiếu cột {significance_column}")
    index = {sector: i for i, sector in enumerate(STANDARD_SECTORS)}
    matrix = np.zeros((len(STANDARD_SECTORS), len(STANDARD_SECTORS)), dtype=int)
    flags = results[significance_column].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes"}
    )
    for (_, row), is_sig in zip(results.iterrows(), flags):
        src, tgt = row.get("source_sector"), row.get("target_sector")
        if is_sig and src in index and tgt in index and src != tgt:
            matrix[index[src], index[tgt]] = 1
    return matrix


def load_price_adjacency(df: pd.DataFrame) -> np.ndarray:
    """Đọc ma trận kề giá (đã vô hướng) thành numpy theo thứ tự STANDARD_SECTORS."""
    frame = df.copy()
    if "source_sector" in frame.columns:
        frame = frame.set_index("source_sector")
    frame = frame.reindex(index=STANDARD_SECTORS, columns=STANDARD_SECTORS)
    matrix = frame.to_numpy(dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0)
    return (matrix != 0).astype(int)


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    """i-j = 1 nếu i->j hoặc j->i; đường chéo = 0."""
    sym = ((matrix + matrix.T) > 0).astype(int)
    np.fill_diagonal(sym, 0)
    return sym


def _offdiag(matrix: np.ndarray) -> np.ndarray:
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return matrix[mask]


def compare(news_sym: np.ndarray, price_sym: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    n = news_sym.shape[0]
    a = _offdiag(news_sym).astype(float)
    b = _offdiag(price_sym).astype(float)

    # Frobenius
    frob_dist = float(np.sqrt(np.sum((a - b) ** 2)))
    frob_sim = float(1.0 - frob_dist / np.sqrt(len(a))) if len(a) else np.nan

    # Jaccard trên cạnh (off-diagonal, undirected)
    inter = int(np.sum((a > 0) & (b > 0)))
    union = int(np.sum((a > 0) | (b > 0)))
    jaccard = float(inter / union) if union else np.nan

    # Tương quan các ô off-diagonal
    if a.std() == 0 or b.std() == 0:
        pear_corr, pear_p = np.nan, np.nan
        spear_corr, spear_p = np.nan, np.nan
    else:
        pear_corr, pear_p = (float(x) for x in pearsonr(a, b))
        spear_corr, spear_p = (float(x) for x in spearmanr(a, b))

    # QAP / Mantel: hoán vị nhãn node của mạng tin tức, tính lại Pearson corr off-diag
    qap_p = np.nan
    qap_perms = 0
    if not np.isnan(pear_corr):
        ge = 0
        for _ in range(N_PERMUTATIONS):
            perm = rng.permutation(n)
            permuted = news_sym[perm][:, perm]
            pa = _offdiag(permuted).astype(float)
            if pa.std() == 0:
                continue
            r, _ = pearsonr(pa, b)
            qap_perms += 1
            if r >= pear_corr:
                ge += 1
        qap_p = float((ge + 1) / (qap_perms + 1)) if qap_perms else np.nan

    return {
        "n_sectors": n,
        "news_edges": int(np.sum(a > 0) // 1),
        "price_edges": int(np.sum(b > 0) // 1),
        "shared_edges": inter,
        "frobenius_distance": frob_dist,
        "frobenius_similarity": frob_sim,
        "jaccard_similarity": jaccard,
        "pearson_corr": pear_corr,
        "pearson_pvalue": pear_p,
        "spearman_corr": spear_corr,
        "spearman_pvalue": spear_p,
        "qap_permutations": qap_perms,
        "qap_pvalue": qap_p,
    }


def to_output_frame(metrics: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(metrics.keys()), "value": list(metrics.values())})


def print_summary(metrics: dict[str, Any]) -> None:
    print("=" * 52)
    print(" SO SÁNH MẠNG TIN TỨC vs MẠNG GIÁ (RQ2)")
    print("=" * 52)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:24s}: {value:.4f}")
        else:
            print(f"  {key:24s}: {value}")
    print("-" * 52)
    qp = metrics.get("qap_pvalue")
    jac = metrics.get("jaccard_similarity")
    if qp is not None and not (isinstance(qp, float) and np.isnan(qp)):
        if qp < 0.05:
            print("  => Hai mạng TƯƠNG ĐỒNG có ý nghĩa (QAP p<0.05): tin tức phản ánh cấu trúc giá.")
        else:
            print("  => Hai mạng KHÁC nhau (QAP p>=0.05): news và price encode chiều thông tin riêng.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--news-results", type=Path, default=DEFAULT_NEWS_RESULTS)
    parser.add_argument("--price-adjacency", type=Path, default=DEFAULT_PRICE_ADJACENCY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--news-significance-column",
        default="significant_raw_05",
        help="Cột significance dùng để dựng cạnh mạng tin tức (raw vì FDR thường rỗng)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        news_results = read_csv(args.news_results)
        price_df = read_csv(args.price_adjacency)

        news_dir = build_news_adjacency(news_results, args.news_significance_column)
        news_sym = symmetrize(news_dir)
        price_sym = load_price_adjacency(price_df)
        np.fill_diagonal(price_sym, 0)

        rng = np.random.default_rng(RANDOM_SEED)
        metrics = compare(news_sym, price_sym, rng)

        output_df = to_output_frame(metrics)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_csv(args.output, index=False, encoding=CSV_ENCODING)

        print_summary(metrics)
        print("comparison written to:", args.output.as_posix())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
