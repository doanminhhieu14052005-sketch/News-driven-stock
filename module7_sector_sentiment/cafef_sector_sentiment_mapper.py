"""Normalize CafeF impact and sector fields for sector sentiment analysis."""

from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT = Path("data/processed/articles_session_mapped.csv")
DEFAULT_OUTPUT = Path("data/processed/cafef_articles_labeled.csv")
DEFAULT_LONG_OUTPUT = Path("data/processed/cafef_article_sector_long.csv")
CSV_ENCODING = "utf-8-sig"

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

STANDARD_SECTORS = list(SECTOR_TICKERS)
TICKER_TO_SECTOR = {
    ticker: sector for sector, tickers in SECTOR_TICKERS.items() for ticker in tickers
}

SECTOR_KEYWORDS = {
    "Banking": ["ngân hàng", "bank"],
    "RealEstate": ["bất động sản", "địa ốc", "nhà ở", "khu công nghiệp", "kcn"],
    "SteelMaterials": [
        "thép",
        "vật liệu",
        "vật liệu xây dựng",
        "hóa chất",
        "cao su",
        "xi măng",
        "nhựa",
    ],
    "Technology": ["công nghệ", "viễn thông", "phần mềm", "điện tử"],
    "ConsumerStaples": [
        "thực phẩm",
        "đồ uống",
        "sữa",
        "bia",
        "hàng tiêu dùng thiết yếu",
        "nông nghiệp",
    ],
    "ConsumerDiscretionary": [
        "bán lẻ",
        "hàng tiêu dùng",
        "vàng bạc",
        "phân phối",
        "ô tô",
        "xe",
    ],
    "Energy": ["dầu khí", "xăng dầu", "khí", "nhiên liệu"],
    "IndustrialLogistics": [
        "logistics",
        "cảng",
        "cảng biển",
        "hàng không",
        "vận tải",
        "giao thông vận tải",
        "sân bay",
        "kho vận",
    ],
    "Healthcare": ["y tế", "dược", "bệnh viện", "chăm sóc sức khỏe"],
    "Utilities": ["điện", "nước", "tiện ích", "năng lượng tái tạo"],
}

MACRO_KEYWORDS = [
    "vĩ mô",
    "kinh tế vĩ mô",
    "lãi suất",
    "tỷ giá",
    "cpi",
    "gdp",
    "thuế",
    "chính sách",
    "thị trường chung",
    "chứng khoán",
    "tài chính",
]

TICKER_COLUMNS = [
    "summary_json.tickers",
    "summary_tickers",
    "tickers",
    "ticker",
    "ticker_raw",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module7_sector_sentiment")


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _normalize_text(value: Any) -> str:
    if _is_missing(value):
        return ""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(character for character in text if not unicodedata.combining(character))
    return re.sub(r"\s+", " ", text)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path.as_posix()}")
    return pd.read_csv(path, encoding=CSV_ENCODING)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding=CSV_ENCODING)


def parse_summary_json(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if _is_missing(value):
        return {}
    text = str(value).strip()
    if not text:
        return {}
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(text)
            return parsed if isinstance(parsed, dict) else {}
        except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
            continue
    return {}


def get_summary_field(row: pd.Series, dotted_column: str, key: str) -> Any:
    if dotted_column in row.index and not _is_missing(row[dotted_column]):
        return row[dotted_column]
    summary = parse_summary_json(row.get("summary_json"))
    return summary.get(key)


def map_sentiment(value: Any) -> tuple[Any, str | None, float, bool]:
    raw = value
    normalized = _normalize_text(value)
    if normalized in {"positive", "tich cuc"}:
        return raw, "Positive", 1.0, True
    if normalized in {"neutral", "trung lap", "trung tinh"}:
        return raw, "Neutral", 0.0, True
    if normalized in {"negative", "tieu cuc"}:
        return raw, "Negative", -1.0, True
    return raw, None, np.nan, False


def coerce_continuous_score(value: Any) -> float | None:
    """Lấy sentiment_score liên tục [-1, 1] từ LLM; None nếu thiếu/không hợp lệ."""
    if _is_missing(value):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(score):
        return None
    return max(-1.0, min(1.0, score))


def parse_ticker_list(value: Any) -> list[str]:
    if _is_missing(value):
        return []
    parsed_value = value
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return []
        parsed_value = None
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed_value = parser(text)
                break
            except (ValueError, SyntaxError, TypeError, json.JSONDecodeError):
                continue
        if parsed_value is None:
            parsed_value = re.findall(r"\b[A-Z]{2,5}\b", text.upper())

    if isinstance(parsed_value, dict):
        parsed_value = list(parsed_value.values())
    if not isinstance(parsed_value, (list, tuple, set)):
        parsed_value = [parsed_value]

    tickers: list[str] = []
    for item in parsed_value:
        if _is_missing(item):
            continue
        ticker = re.sub(r"[^A-Z0-9]", "", str(item).upper())
        if ticker and ticker not in tickers:
            tickers.append(ticker)
    return tickers


def get_ticker_raw(row: pd.Series) -> Any:
    for column in TICKER_COLUMNS:
        if column in row.index and not _is_missing(row[column]):
            return row[column]
    summary = parse_summary_json(row.get("summary_json"))
    return summary.get("tickers")


def _contains_keyword(normalized_text: str, keyword: str) -> bool:
    """Khớp keyword theo BIÊN TỪ (word boundary) để tránh false positive khi bỏ dấu.
    Vd 'khi' (khí) không còn dính trong 'khich'/'khien', 'xe'/'nuoc'/'dien' không dính
    trong từ dài hơn."""
    kw = _normalize_text(keyword)
    if not kw:
        return False
    return re.search(rf"(?<!\w){re.escape(kw)}(?!\w)", normalized_text) is not None


def match_sector_keywords(text: Any) -> list[str]:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return []

    matched: list[str] = []
    for sector, keywords in SECTOR_KEYWORDS.items():
        for keyword in keywords:
            if _contains_keyword(normalized_text, keyword):
                matched.append(sector)
                break
    return matched


def is_macro_text(text: Any) -> bool:
    normalized_text = _normalize_text(text)
    if not normalized_text:
        return False
    return any(_contains_keyword(normalized_text, keyword) for keyword in MACRO_KEYWORDS)


def build_fallback_text(row: pd.Series) -> str:
    pieces = [
        row.get("title"),
        row.get("category"),
        row.get("summary_json"),
    ]
    return " ".join(str(piece) for piece in pieces if not _is_missing(piece))


def map_sector(row: pd.Series) -> dict[str, Any]:
    sector_raw = get_summary_field(row, "summary_json.sector", "sector")
    ticker_raw = get_ticker_raw(row)
    ticker_list = parse_ticker_list(ticker_raw)
    ticker_sectors = []
    for ticker in ticker_list:
        sector = TICKER_TO_SECTOR.get(ticker)
        if sector and sector not in ticker_sectors:
            ticker_sectors.append(sector)

    if ticker_sectors:
        method = "ticker" if len(ticker_sectors) == 1 else "multi_ticker"
        confidence = 1.0 if len(ticker_sectors) == 1 else 0.95
        primary_sector = ticker_sectors[0]
        return {
            "sector_raw": sector_raw,
            "ticker_raw": ticker_raw,
            "ticker_list": ticker_list,
            "sector_labels": ticker_sectors,
            "primary_sector": primary_sector,
            "sector_label": primary_sector,
            "sector_weight": 1.0 / len(ticker_sectors),
            "sector_mapping_method": method,
            "sector_mapping_confidence": confidence,
            "is_sector_mapped": True,
            "is_macro": False,
        }

    sector_matches = match_sector_keywords(sector_raw)
    if sector_matches:
        primary_sector = sector_matches[0]
        return {
            "sector_raw": sector_raw,
            "ticker_raw": ticker_raw,
            "ticker_list": ticker_list,
            "sector_labels": sector_matches,
            "primary_sector": primary_sector,
            "sector_label": primary_sector,
            "sector_weight": 1.0 / len(sector_matches),
            "sector_mapping_method": "keyword",
            "sector_mapping_confidence": 0.8,
            "is_sector_mapped": True,
            "is_macro": False,
        }

    if is_macro_text(sector_raw):
        return {
            "sector_raw": sector_raw,
            "ticker_raw": ticker_raw,
            "ticker_list": ticker_list,
            "sector_labels": [],
            "primary_sector": "MarketMacro",
            "sector_label": "MarketMacro",
            "sector_weight": np.nan,
            "sector_mapping_method": "macro",
            "sector_mapping_confidence": 0.0,
            "is_sector_mapped": False,
            "is_macro": True,
        }

    fallback_text = build_fallback_text(row)
    fallback_matches = match_sector_keywords(fallback_text)
    if fallback_matches:
        primary_sector = fallback_matches[0]
        return {
            "sector_raw": sector_raw,
            "ticker_raw": ticker_raw,
            "ticker_list": ticker_list,
            "sector_labels": fallback_matches,
            "primary_sector": primary_sector,
            "sector_label": primary_sector,
            "sector_weight": 1.0 / len(fallback_matches),
            "sector_mapping_method": "keyword_fallback",
            "sector_mapping_confidence": 0.6,
            "is_sector_mapped": True,
            "is_macro": False,
        }

    if is_macro_text(fallback_text):
        return {
            "sector_raw": sector_raw,
            "ticker_raw": ticker_raw,
            "ticker_list": ticker_list,
            "sector_labels": [],
            "primary_sector": "MarketMacro",
            "sector_label": "MarketMacro",
            "sector_weight": np.nan,
            "sector_mapping_method": "macro",
            "sector_mapping_confidence": 0.0,
            "is_sector_mapped": False,
            "is_macro": True,
        }

    return {
        "sector_raw": sector_raw,
        "ticker_raw": ticker_raw,
        "ticker_list": ticker_list,
        "sector_labels": [],
        "primary_sector": "Unmapped",
        "sector_label": "Unmapped",
        "sector_weight": np.nan,
        "sector_mapping_method": "unmapped",
        "sector_mapping_confidence": 0.0,
        "is_sector_mapped": False,
        "is_macro": False,
    }


def build_article_output(df: pd.DataFrame) -> pd.DataFrame:
    if "trade_date" not in df.columns:
        raise RuntimeError("Input articles file is missing trade_date; Module 8 needs trade_date")

    result = df.copy()
    impact_values = result.apply(
        lambda row: get_summary_field(row, "summary_json.impact", "impact"), axis=1
    )
    sentiment = impact_values.apply(map_sentiment)
    sentiment_df = pd.DataFrame(
        sentiment.tolist(),
        columns=[
            "impact_raw",
            "sentiment_label",
            "sentiment_score_impact",
            "is_sentiment_mapped",
        ],
        index=result.index,
    )
    # Ưu tiên sentiment_score LIÊN TỤC từ LLM (-1..1); fallback điểm rời rạc từ impact
    continuous_score = result.apply(
        lambda row: coerce_continuous_score(
            get_summary_field(row, "summary_json.sentiment_score", "sentiment_score")
        ),
        axis=1,
    )
    sentiment_df["sentiment_score"] = continuous_score.where(
        continuous_score.notna(), sentiment_df["sentiment_score_impact"]
    )
    sentiment_df["sentiment_score_source"] = np.where(
        continuous_score.notna(), "llm_continuous", "impact_fallback"
    )

    sector_df = pd.DataFrame(result.apply(map_sector, axis=1).tolist(), index=result.index)
    result = pd.concat([result, sentiment_df, sector_df], axis=1)
    result["_sector_label_list"] = result["sector_labels"]
    result["ticker_list"] = result["ticker_list"].apply(_json_dumps)
    result["sector_labels"] = result["sector_labels"].apply(_json_dumps)
    return result


def build_long_output(article_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in article_df.iterrows():
        labels = row["_sector_label_list"]
        if not row["is_sector_mapped"] or not labels:
            continue
        sector_weight = 1.0 / len(labels)
        for label in labels:
            if label not in STANDARD_SECTORS:
                continue
            record = row.drop(labels=["_sector_label_list"]).to_dict()
            record["sector_label"] = label
            record["sector_weight"] = sector_weight
            rows.append(record)

    if not rows:
        return pd.DataFrame(columns=[column for column in article_df.columns if column != "_sector_label_list"])
    return pd.DataFrame(rows)


def validate_outputs(article_df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    scores = pd.to_numeric(article_df["sentiment_score"], errors="coerce").dropna()
    out_of_range = scores[(scores < -1.0) | (scores > 1.0)]
    if len(out_of_range):
        raise RuntimeError(
            f"sentiment_score ngoài [-1, 1]: {sorted(out_of_range.unique().tolist())[:5]}"
        )

    if not long_df.empty:
        invalid_long_sectors = sorted(set(long_df["sector_label"]) - set(STANDARD_SECTORS))
        if invalid_long_sectors:
            raise RuntimeError(f"Invalid sector_label values in long output: {invalid_long_sectors}")
        forbidden = {"MarketMacro", "Unmapped"}.intersection(set(long_df["sector_label"]))
        if forbidden:
            raise RuntimeError(f"Long output contains non-sector labels: {sorted(forbidden)}")

    null_trade_dates = int(pd.to_datetime(article_df["trade_date"], errors="coerce").isna().sum())
    if null_trade_dates:
        logger.warning("Article output contains %d null trade_date values", null_trade_dates)


def print_summary(input_df: pd.DataFrame, article_df: pd.DataFrame, long_df: pd.DataFrame) -> None:
    print("input shape:", input_df.shape)
    print("article-level output shape:", article_df.shape)
    print("long output shape:", long_df.shape)
    print("sentiment_score distribution:")
    print(article_df["sentiment_score"].describe().to_string())
    if "sentiment_score_source" in article_df.columns:
        print("sentiment_score source:")
        print(article_df["sentiment_score_source"].value_counts(dropna=False).to_string())
    print("sector_labels distribution:")
    print(article_df["sector_label"].value_counts(dropna=False).to_string())
    print("sector_mapping_method distribution:")
    print(article_df["sector_mapping_method"].value_counts(dropna=False).to_string())
    print("number of unmapped articles:", int((article_df["sector_label"] == "Unmapped").sum()))
    print("number of macro articles:", int(article_df["is_macro"].sum()))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--long-output", type=Path, default=DEFAULT_LONG_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        input_df = read_csv(args.input)
        article_df = build_article_output(input_df)
        long_df = build_long_output(article_df)
        article_to_write = article_df.drop(columns=["_sector_label_list"])
        validate_outputs(article_to_write, long_df)
        write_csv(article_to_write, args.output)
        write_csv(long_df, args.long_output)
        print_summary(input_df, article_to_write, long_df)
        print("article output written to:", args.output.as_posix())
        print("long output written to:", args.long_output.as_posix())
        return 0
    except (FileNotFoundError, RuntimeError, ValueError, pd.errors.ParserError) as exc:
        logger.error("%s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
