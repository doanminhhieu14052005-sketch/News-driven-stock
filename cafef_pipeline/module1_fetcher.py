"""
Module 1: URL Fetcher & Deduplication
- Crawl danh mục CafeF, lấy [title, url, published_at]
- Dedup bằng MongoDB url_hash (SHA256)
- Hỗ trợ resumable crawl: nhớ trang đã cào, lần sau chạy tiếp
- Trả về queue các URL chưa xử lý
"""

import hashlib
import json
import os
import random
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup
from pymongo import ReturnDocument

from config import (
    CAFEF_CATEGORIES,
    REQUEST_DELAY, REQUEST_HEADERS, SCRAPE_DEPTH,
    MAX_PAGES_PER_RUN, CRAWL_PROGRESS_FILE, CRAWL_LOOKBACK_DAYS
)
from module4_storage import get_collection
from utils import parse_published_at

logger = logging.getLogger(__name__)


def _parse_pub_dt(s: str) -> Optional[datetime]:
    """Parse published_at về datetime. Hỗ trợ cả ISO và DD/MM/YYYY (day-first)."""
    return parse_published_at(s)


# ── Database helpers (MongoDB) ──────────────────────────────────

def hash_url(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()


def is_seen(url_hash: str) -> bool:
    col = get_collection()
    return col.find_one({"url_hash": url_hash}, {"_id": 1}) is not None


def mark_seen(item: dict) -> None:
    """Insert URL mới với status=pending. Bỏ qua nếu đã tồn tại."""
    col = get_collection()
    col.update_one(
        {"url_hash": item["url_hash"]},
        {"$setOnInsert": {
            "url_hash":     item["url_hash"],
            "source_url":   item["url"],
            "title":        item.get("title", ""),
            "published_at": parse_published_at(item.get("published_at")),
            "category":     item.get("category", ""),
            "raw_text":     "",
            "summary_json": None,
            "status":       "pending",
            "retry_count":  0,
            "scrape_method": "",
            "created_at":   datetime.now(timezone.utc),
        }},
        upsert=True,
    )


def get_pending_urls(limit: int = 20) -> list[dict]:
    col = get_collection()
    # Lấy cả "pending", "processing" và "failed" (chưa quá 3 lần) để retry
    cursor = col.find(
        {"status": {"$in": ["pending", "processing", "failed"]}, "retry_count": {"$lt": 3}},
        {"url_hash": 1, "source_url": 1, "title": 1,
         "published_at": 1, "category": 1, "_id": 0}
    ).sort("created_at", -1).limit(limit)

    return [
        {"url_hash": doc["url_hash"], "url": doc["source_url"],
         "title": doc.get("title", ""), "published_at": doc.get("published_at", ""),
         "category": doc.get("category", "")}
        for doc in cursor
    ]


def claim_next_pending(stale_minutes: int = 15) -> Optional[dict]:
    """CLAIM nguyên tử 1 bài để xử lý — an toàn khi nhiều máy chạy song song.

    Dùng find_one_and_update: mỗi bài chỉ đúng 1 worker nhận được (đặt status=processing
    + claimed_at trong cùng 1 thao tác nguyên tử). Trả None khi hàng đợi rỗng.

    Nhận: bài 'pending'/'failed' (retry_count<3), HOẶC bài 'processing' bị treo quá
    stale_minutes (máy khác chết giữa chừng) để không kẹt vĩnh viễn.
    """
    col = get_collection()
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(minutes=stale_minutes)
    doc = col.find_one_and_update(
        {
            "retry_count": {"$lt": 3},
            "$or": [
                {"status": {"$in": ["pending", "failed"]}},
                {"status": "processing", "claimed_at": {"$lt": stale_cutoff}},
            ],
        },
        {"$set": {"status": "processing", "claimed_at": now}},
        projection={"url_hash": 1, "source_url": 1, "title": 1,
                    "published_at": 1, "category": 1, "_id": 0},
        return_document=ReturnDocument.AFTER,
    )
    if not doc:
        return None
    return {
        "url_hash": doc["url_hash"], "url": doc["source_url"],
        "title": doc.get("title", ""), "published_at": doc.get("published_at", ""),
        "category": doc.get("category", ""),
    }


def update_status(url_hash: str, status: str) -> None:
    """Cập nhật status. Nếu failed thì tăng retry_count."""
    col = get_collection()
    update: dict = {"$set": {"status": status}}
    if status == "failed":
        update["$inc"] = {"retry_count": 1}
    col.update_one({"url_hash": url_hash}, update)


# ── Crawl Progress (Resumable) ──────────────────────────────────

def _load_progress() -> dict:
    """Đọc tiến trình cào từ file JSON. Trả về {} nếu chưa có."""
    if os.path.exists(CRAWL_PROGRESS_FILE):
        try:
            with open(CRAWL_PROGRESS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            logger.warning("Không đọc được crawl_progress.json, bắt đầu lại từ đầu.")
    return {}


def _save_progress(progress: dict) -> None:
    """Lưu tiến trình cào vào file JSON."""
    os.makedirs(os.path.dirname(CRAWL_PROGRESS_FILE), exist_ok=True)
    with open(CRAWL_PROGRESS_FILE, "w", encoding="utf-8") as f:
        json.dump(progress, f, ensure_ascii=False, indent=2)
    logger.info(f"💾 Đã lưu tiến trình: {progress}")


def reset_progress() -> None:
    """Reset tiến trình, lần chạy sau sẽ bắt đầu từ trang 1."""
    if os.path.exists(CRAWL_PROGRESS_FILE):
        os.remove(CRAWL_PROGRESS_FILE)
        logger.info("🔄 Đã reset tiến trình crawl.")


# ── Fetcher ──────────────────────────────────────────────────────

def _safe_get(url: str) -> Optional[requests.Response]:
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


def _parse_article_list(html: str, category: str) -> list[dict]:
    soup = BeautifulSoup(html, "html.parser")
    articles = []

    for item in soup.select("div.tlitem, div.item-news, div.tinmoi li, div.firstitem, div.big"):
        
        # --- Theo dõi nguồn gốc ---
        tag_name = item.name
        tag_classes = item.get("class")
        
        if tag_classes:
            source_info = f"{tag_name} class: {' '.join(tag_classes)}"
        else:
            source_info = f"{tag_name} (nằm trong tinmoi)"
        # ----------------------------------------

        a_tag = item.select_one("h3 a, h2 a, .title a")
        if not a_tag:
            a_tag = item.select_one("a")

        if not a_tag:
            continue

        url = a_tag.get("href", "")
        if not url.startswith("http"):
            url = "https://cafef.vn" + url

        title = a_tag.get_text(strip=True)
        time_tag = item.select_one("span.time, span.date, time, p.time")
        published_at = time_tag.get_text(strip=True) if time_tag else ""

        articles.append({
            "url": url,
            "title": title,
            "published_at": published_at,
            "category": category,
            "url_hash": hash_url(url),
            "source_box": source_info
        })

    return articles


def fetch_new_urls(depth: int = SCRAPE_DEPTH, max_pages: int | None = MAX_PAGES_PER_RUN) -> list[dict]:
    """
    Fetch URLs mới từ CafeF, hỗ trợ resumable crawl.
    
    - depth:     Tổng số trang tối đa muốn cào (mục tiêu cuối cùng)
    - max_pages: Số trang tối đa cho LẦN CHẠY NÀY (None = chạy hết depth)
    
    Tiến trình được lưu vào file JSON, lần chạy sau tự động tiếp tục.
    """
    new_items = []
    progress = _load_progress()

    # Mốc thời gian: chỉ cào bài mới hơn cutoff (mục tiêu ~1 năm)
    cutoff = (datetime.now() - timedelta(days=CRAWL_LOOKBACK_DAYS)) if CRAWL_LOOKBACK_DAYS else None
    if cutoff:
        logger.info(f"🗓️  Cào tới mốc: {cutoff:%d/%m/%Y} (lookback {CRAWL_LOOKBACK_DAYS} ngày)")

    for cat_name, info in CAFEF_CATEGORIES.items():
        # Đọc trang đã cào lần trước (0 = chưa cào)
        start_page = progress.get(cat_name, 0) + 1
        end_page = min(depth, start_page + max_pages - 1) if max_pages else depth

        if start_page > depth:
            logger.info(f"--- {cat_name.upper()}: Đã cào hết {depth} trang, bỏ qua ---")
            continue

        logger.info(
            f"--- Đang quét danh mục: {cat_name.upper()} | "
            f"Trang {start_page} → {end_page} (tổng mục tiêu: {depth}) ---"
        )

        last_page_done = progress.get(cat_name, 0)
        reached_cutoff = False

        for page in range(start_page, end_page + 1):
            if page == 1:
                target_url = info["url"]
            else:
                target_url = f"https://cafef.vn/timelinelist/{info['api_id']}/{page}.chn"

            logger.info(f"Đang quét Trang {page}/{end_page}: {target_url}")
            resp = _safe_get(target_url)
            if not resp:
                continue

            articles = _parse_article_list(resp.text, cat_name)

            page_has_recent = False
            for article in articles:
                pub = _parse_pub_dt(article["published_at"])
                # Bỏ qua bài cũ hơn mốc 1 năm
                if cutoff and pub and pub < cutoff:
                    reached_cutoff = True
                    continue
                page_has_recent = True
                if not is_seen(article["url_hash"]):
                    mark_seen(article)
                    new_items.append(article)
                    logger.debug(f"New [{article['source_box']}]: {article['title'][:50]}")

            last_page_done = page
            time.sleep(random.uniform(*REQUEST_DELAY))

            # Dừng danh mục khi cả trang đều cũ hơn mốc thời gian
            if cutoff and reached_cutoff and not page_has_recent:
                logger.info(
                    f"--- {cat_name.upper()}: Đã chạm mốc {CRAWL_LOOKBACK_DAYS} ngày tại trang {page}. "
                    f"Hoàn tất cào danh mục này. ---"
                )
                last_page_done = depth  # đánh dấu hoàn tất -> lần chạy sau bỏ qua
                break

        # Cập nhật tiến trình cho danh mục này
        progress[cat_name] = last_page_done

    # Lưu tiến trình sau khi hoàn thành
    _save_progress(progress)

    # Kiểm tra xem đã cào hết chưa (mọi danh mục chạm mốc thời gian / SCRAPE_DEPTH)
    all_done = all(progress.get(cat, 0) >= depth for cat in CAFEF_CATEGORIES)
    if all_done:
        # Đã backfill đủ ~1 năm. KHÔNG reset để giữ dataset ổn định + tránh cào lại từ đầu mỗi chu kỳ.
        # Muốn cào lại bài mới: gọi reset_progress() thủ công.
        logger.info("🎉 ĐÃ CÀO ĐỦ ~1 NĂM cho tất cả danh mục. Giữ nguyên tiến trình (không reset).")

    logger.info(f"Tổng kết: Tìm thấy {len(new_items)} link mới trong lần chạy này.")
    return new_items