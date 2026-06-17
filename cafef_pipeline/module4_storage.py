"""
Module 4: Storage (MongoDB)
- Quản lý kết nối MongoDB (singleton)
- Lưu article đã xử lý
- Cung cấp get_collection() cho các module khác
"""

import logging
from datetime import datetime, timezone

from pymongo import MongoClient, DESCENDING
from pymongo.collection import Collection

from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

logger = logging.getLogger(__name__)

# ── Singleton connection ─────────────────────────────────────────

_client: MongoClient | None = None
_collection: Collection | None = None


def get_collection() -> Collection:
    """Trả về collection articles, tạo index 1 lần duy nhất."""
    global _client, _collection
    if _collection is None:
        _client = MongoClient(MONGO_URI)
        db = _client[MONGO_DB]
        _collection = db[MONGO_COLLECTION]
        # Indexes
        _collection.create_index([("url_hash", 1)], unique=True)
        _collection.create_index([("published_at", DESCENDING)])
        _collection.create_index([("summary_json.tickers", 1)])
        _collection.create_index([("summary_json.impact", 1)])
        _collection.create_index([("status", 1)])
    return _collection


# ── Save ─────────────────────────────────────────────────────────

def save_article(article: dict) -> bool:
    """Cập nhật 1 article đã scrape + summarize vào MongoDB."""
    col = get_collection()
    try:
        col.update_one(
            {"url_hash": article["url_hash"]},
            {"$set": {
                "source_url":    article.get("url", ""),
                "title":         article.get("title", ""),
                "category":      article.get("category", ""),
                "published_at":  article.get("published_at", ""),
                "raw_text":      article.get("raw_text", ""),
                "summary_json":  article.get("summary_json"),
                "status":        article.get("status", "done"),
                "scrape_method": article.get("method", ""),
                "updated_at":    datetime.now(timezone.utc),
            }},
            upsert=True,
        )
        return True
    except Exception as e:
        logger.error(f"DB save error: {e}")
        return False


def save_articles(articles: list[dict]) -> int:
    """Lưu batch, trả về số bản ghi thành công."""
    if not articles:
        return 0
    saved = sum(1 for a in articles if save_article(a))
    logger.info(f"Saved {saved}/{len(articles)} articles to MongoDB")
    return saved