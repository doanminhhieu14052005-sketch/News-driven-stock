"""Export CafeF articles from MongoDB to the Module 6 input CSV."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_OUTPUT = Path("data/raw/articles.csv")
DEFAULT_MONGO_DB_NAME = "cafef_news"
DEFAULT_MONGO_COLLECTION = "articles"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("module6_export_cafef")


def _json_string(value: Any) -> str | None:
    """Serialize nested MongoDB values without leaking credentials."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _article_id(document: dict[str, Any]) -> str | None:
    value = document.get("article_id", document.get("_id"))
    return None if value is None else str(value)


def _normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    summary_json = document.get("summary_json")
    if not isinstance(summary_json, dict):
        summary_json = {}

    return {
        "article_id": _article_id(document),
        "source": document.get("source", "CafeF"),
        "title": document.get("title"),
        "url": document.get("url") or document.get("source_url"),
        "published_at": document.get("published_at"),
        "category": document.get("category"),
        "summary_json": _json_string(document.get("summary_json")),
        "summary_json.impact": summary_json.get("impact"),
        "summary_json.sector": summary_json.get("sector"),
        "summary_json.tickers": _json_string(summary_json.get("tickers")),
    }


def export_cafef_articles(output: Path, limit: int | None = None) -> pd.DataFrame:
    """Read CafeF articles with published_at from MongoDB and write a CSV."""
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        logger.warning("python-dotenv is not installed; reading environment only")

    mongo_uri = os.getenv("MONGO_URI")
    mongo_db_name = os.getenv("MONGO_DB_NAME") or DEFAULT_MONGO_DB_NAME
    mongo_collection = os.getenv("MONGO_COLLECTION") or DEFAULT_MONGO_COLLECTION

    if not mongo_uri:
        raise RuntimeError("MONGO_URI environment variable is not set")

    from pymongo import MongoClient

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    try:
        collection = client[mongo_db_name][mongo_collection]
        missing_published_at = collection.count_documents(
            {"$or": [{"published_at": {"$exists": False}}, {"published_at": None}]}
        )
        query = {
            "published_at": {"$exists": True, "$ne": None},
            "$or": [
                {"source": {"$regex": "^cafef$", "$options": "i"}},
                {"source": {"$exists": False}},
            ],
        }
        cursor = collection.find(query).sort("published_at", 1)
        if limit is not None:
            cursor = cursor.limit(limit)

        records = [_normalize_document(document) for document in cursor]
    finally:
        client.close()

    df = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)

    logger.info("Exported CafeF articles: %d", len(df))
    logger.info("Articles missing published_at in collection: %d", missing_published_at)
    logger.info("Output written to: %s", output)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_cafef_articles(args.output, args.limit)
        return 0
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    except Exception as exc:
        logger.error("CafeF export failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())
