"""
Chạy MỘT chu kỳ pipeline rồi thoát — dùng cho Windows Task Scheduler (chạy daily trên máy local).

Luồng: cào URL mới (Module 1) -> xử lý pending bằng Ollama (Module 2+3) -> lưu MongoDB.
Khác orchestrator.py ở chỗ KHÔNG vào vòng lặp scheduler (chạy 1 lần rồi exit) nên Task
Scheduler quản lý được.

Yêu cầu: Ollama đang chạy (LLM_BACKEND=ollama trong .env).

Chạy tay:
    python run_daily.py
"""

import logging

from config import BATCH_SIZE
from module1_fetcher import fetch_new_urls, get_pending_urls
from module2_scraper import process_pending_articles

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_daily")


def main() -> None:
    logger.info("=== RUN DAILY: bắt đầu 1 chu kỳ ===")
    new_urls = fetch_new_urls()
    logger.info("M1: %d URL mới được thêm vào hàng đợi", len(new_urls))

    pending = get_pending_urls(limit=BATCH_SIZE)
    if not pending:
        logger.info("Không có bài pending để xử lý. Kết thúc.")
        return
    process_pending_articles(pending)
    logger.info("=== RUN DAILY: hoàn tất ===")


if __name__ == "__main__":
    main()
