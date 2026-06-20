"""
Drain backlog — rút cạn các bài 'pending' đang tồn trong MongoDB.

AN TOÀN KHI NHIỀU MÁY CHẠY SONG SONG: mỗi bài được CLAIM nguyên tử
(find_one_and_update) nên mỗi máy nhận một tập bài khác nhau, không trùng,
không tốn quota LLM gấp đôi. Máy nào chết giữa chừng thì bài 'processing'
treo sẽ được máy khác nhận lại sau 15 phút.

KHÔNG cào URL mới (bỏ qua Module 1) — chỉ xử lý backlog có sẵn.
Backend LLM lấy từ .env (LLM_BACKEND): mỗi người nên đặt GROQ key riêng
của mình (org riêng = quota riêng, không kéo nhau bị chặn), hoặc dùng Ollama.

Cách chạy trên MỖI máy (sau khi `conda activate tf-gpu`):
    python drain_backlog.py
Dừng an toàn bất cứ lúc nào bằng Ctrl+C — bài đã xong đã nằm trong DB.
"""

import json
import logging
import os
from datetime import date

from config import LLM_BACKEND
from module1_fetcher import claim_next_pending, update_status
from module2_scraper import scrape_article
from module3_summarizer import summarize_single
from module4_storage import get_collection, save_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("drain")


def count_remaining() -> int:
    col = get_collection()
    return col.count_documents(
        {"status": {"$in": ["pending", "failed"]}, "retry_count": {"$lt": 3}}
    )


def drain(output_dir: str = "data") -> None:
    os.makedirs(output_dir, exist_ok=True)
    out_file = os.path.join(output_dir, f"raw_articles_{date.today().isoformat()}.jsonl")

    remaining = count_remaining()
    logger.info(f"🚀 BẮT ĐẦU DRAIN | backend={LLM_BACKEND} | còn ~{remaining} bài trong hàng đợi")

    done = fail = processed = 0
    with open(out_file, "a", encoding="utf-8") as f:
        while True:
            item = claim_next_pending()
            if item is None:
                logger.info("✅ Hàng đợi rỗng — không còn bài để claim. Kết thúc.")
                break

            processed += 1
            url = item["url"]
            url_hash = item["url_hash"]
            try:
                scraped = scrape_article(url)
                if not scraped:
                    update_status(url_hash, "failed")
                    fail += 1
                    logger.warning(f"[{processed}] ❌ Scrape fail: {url[:60]}")
                    continue

                full = {**item, **scraped}
                full = summarize_single(full)        # gắn summary_json (có sentiment_score)
                save_articles([full])                # lưu Mongo ngay

                f.write(json.dumps(full, ensure_ascii=False, default=str) + "\n")
                f.flush()

                status = "done" if full.get("status") == "done" else "failed"
                update_status(url_hash, status)
                if status == "done":
                    done += 1
                else:
                    fail += 1

                if processed % 50 == 0:
                    logger.info(f"[{processed}] done={done} fail={fail} | còn ~{count_remaining()}")

            except Exception as e:
                logger.error(f"[{processed}] 💥 {url[:60]} — {e}")
                update_status(url_hash, "failed")
                fail += 1

    logger.info(f"🏁 HOÀN TẤT máy này: {done} done, {fail} fail / {processed} đã claim")


if __name__ == "__main__":
    try:
        drain()
    except KeyboardInterrupt:
        logger.info("⏹️ Dừng bởi người dùng (Ctrl+C). Bài đã xong vẫn an toàn trong DB.")
