"""
Drain theo CATEGORY + giới hạn số bài.

Xử lý các bài 'pending' của MỘT chuyên mục (mặc định 'bat_dong_san'), tối đa --limit bài.
Dùng cùng cơ chế claim nguyên tử như drain_backlog (an toàn khi chạy song song nhiều máy):
mỗi bài chỉ một worker nhận.

Cách chạy (sau khi `conda activate tf-gpu`, trong thư mục cafef_pipeline):
    python drain_category.py                          # 60 bài bất động sản
    python drain_category.py --category bat_dong_san --limit 70
    python drain_category.py --category chung_khoan --limit 50

Category hợp lệ (khớp config.CAFEF_CATEGORIES):
    vi_mo | chung_khoan | doanh_nghiep | tai_chinh_nh | bat_dong_san | hang_hoa
"""

import argparse
import json
import logging
import os
from datetime import date, datetime, timezone, timedelta
from typing import Optional

from pymongo import ReturnDocument

from config import LLM_BACKEND
from module1_fetcher import update_status
from module2_scraper import scrape_article
from module3_summarizer import summarize_single
from module4_storage import get_collection, save_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drain_category")


def claim_next_in_category(category: str, stale_minutes: int = 15) -> Optional[dict]:
    """Claim nguyên tử 1 bài pending/failed/stale của đúng category này."""
    col = get_collection()
    now = datetime.now(timezone.utc)
    stale_cutoff = now - timedelta(minutes=stale_minutes)
    doc = col.find_one_and_update(
        {
            "category": category,
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


def count_pending_in_category(category: str) -> int:
    col = get_collection()
    return col.count_documents(
        {"category": category, "status": {"$in": ["pending", "failed"]}, "retry_count": {"$lt": 3}}
    )


def drain(category: str, limit: int, output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"raw_articles_{date.today().isoformat()}.jsonl")

    available = count_pending_in_category(category)
    logger.info(
        f"🚀 DRAIN category='{category}' | backend={LLM_BACKEND} | "
        f"mục tiêu {limit} bài | còn ~{available} pending trong chuyên mục"
    )

    done = fail = processed = 0
    with open(out_file, "a", encoding="utf-8") as f:
        while processed < limit:
            item = claim_next_in_category(category)
            if item is None:
                logger.info("✅ Hết bài pending trong chuyên mục — dừng sớm.")
                break

            processed += 1
            url, url_hash = item["url"], item["url_hash"]
            try:
                scraped = scrape_article(url)
                if not scraped:
                    update_status(url_hash, "failed")
                    fail += 1
                    logger.warning(f"[{processed}/{limit}] ❌ Scrape fail: {url[:60]}")
                    continue

                full = {**item, **scraped}
                full = summarize_single(full)
                save_articles([full])

                f.write(json.dumps(full, ensure_ascii=False, default=str) + "\n")
                f.flush()

                status = "done" if full.get("status") == "done" else "failed"
                update_status(url_hash, status)
                if status == "done":
                    done += 1
                else:
                    fail += 1
                logger.info(
                    f"[{processed}/{limit}] {'✅' if status == 'done' else '⚠️'} {status.upper()} | {url[:60]}"
                )
            except Exception as e:
                logger.error(f"[{processed}/{limit}] 💥 {url[:60]} — {e}")
                update_status(url_hash, "failed")
                fail += 1

    logger.info(f"🏁 HOÀN TẤT: {done} done, {fail} fail / {processed} đã xử lý (category='{category}')")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--category", default="bat_dong_san", help="chuyên mục cần cày")
    parser.add_argument("--limit", type=int, default=60, help="số bài tối đa (vd 50-70)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        drain(args.category, args.limit)
    except KeyboardInterrupt:
        logger.info("⏹️ Dừng bởi người dùng (Ctrl+C). Bài đã xong vẫn an toàn trong DB.")
