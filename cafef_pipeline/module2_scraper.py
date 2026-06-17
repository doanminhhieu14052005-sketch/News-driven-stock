"""
Module 2: Scraper & Cleaner
- Tải HTML từng bài viết
- Bóc tách nội dung sạch nhắm selector .knc-content
- Fallback sang trafilatura nếu selector thất bại
- Trả về raw_text sạch
"""

import random
import re
import time
import logging
from typing import Optional

import json
import os
from tqdm import tqdm
import requests
import trafilatura
from bs4 import BeautifulSoup

from config import REQUEST_DELAY, REQUEST_HEADERS

logger = logging.getLogger(__name__)

# Regex loại bỏ text nhiễu đặc trưng CafeF
NOISE_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r"xem thêm.{0,50}",
        r"đọc thêm.{0,50}",
        r"cùng chuyên mục.{0,50}",
        r"theo dõi.{0,30}cafef",
        r"nguồn:.{0,80}",
        r"tags?:.{0,200}",
        r"\[.*?\]",           # [Ảnh minh họa], [Video]...
        r"<[^>]+>",           # HTML tags sót lại
    ]
]

MIN_CONTENT_LENGTH = 500  # Tăng ngưỡng để loại bài ảnh/video chất lượng thấp


def _clean_text(text: str) -> str:
    """Xóa noise, chuẩn hóa whitespace."""
    for pattern in NOISE_PATTERNS:
        text = pattern.sub(" ", text)
    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s{2,}", "\n", text)
    return text.strip()


def _extract_with_bs4(html: str) -> Optional[str]:
    """Phương pháp 1: Nhắm thẳng vào selector CafeF."""
    soup = BeautifulSoup(html, "html.parser")

    # Xóa script/style/aside trước
    for tag in soup(["script", "style", "aside", "nav", "footer", "header"]):
        tag.decompose()

    # Selector chính của CafeF
    content_div = (
        soup.select_one(".knc-content")
        or soup.select_one(".detail-content")
        or soup.select_one(".article-body")
        or soup.select_one("div.nd_detail")
    )

    if not content_div:
        return None

    # Xóa quảng cáo inline
    for ad in content_div.select(".social-share, .ads, .related-news, figure"):
        ad.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in content_div.find_all("p")]
    return "\n".join(paragraphs)


def _extract_with_trafilatura(html: str, url: str) -> Optional[str]:
    """Phương pháp 2: Fallback dùng trafilatura."""
    return trafilatura.extract(
        html,
        url=url,
        include_comments=False,
        include_tables=False,
        no_fallback=False,
        favor_precision=True,
    )


def scrape_article(url: str) -> Optional[dict]:
    """
    Trả về dict:
    {
        "url": str,
        "raw_text": str,
        "method": "bs4" | "trafilatura"
    }
    Trả None nếu không lấy được nội dung.
    """
    # Retry với exponential backoff
    html = None
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
            resp.raise_for_status()
            html = resp.text
            break
        except requests.RequestException as e:
            if attempt == 2:
                logger.warning(f"Scrape failed after 3 attempts: {url}: {e}")
                return None
            wait = 2 ** attempt
            logger.debug(f"Scrape retry {attempt+1}/3 for {url[:50]}, waiting {wait}s")
            time.sleep(wait)

    # Thử BS4 trước (chính xác hơn với CafeF)
    text = _extract_with_bs4(html)
    method = "bs4"

    # Fallback trafilatura nếu BS4 thất bại hoặc quá ngắn
    if not text or len(text) < MIN_CONTENT_LENGTH:
        text = _extract_with_trafilatura(html, url)
        method = "trafilatura"

    if not text or len(text) < MIN_CONTENT_LENGTH:
        logger.warning(f"No content extracted from {url}")
        return None

    clean = _clean_text(text)
    logger.info(f"Scraped ({method}) {len(clean)} chars: {url[:60]}")

    time.sleep(random.uniform(*REQUEST_DELAY))

    return {
        "url": url,
        "raw_text": clean,
        "method": method,
    }


def scrape_batch(items: list[dict]) -> list[dict]:
    """Scrape nhiều bài, giữ lại metadata từ Module 1."""
    results = []
    for item in items:
        scraped = scrape_article(item["url"])
        if scraped:
            results.append({**item, **scraped})
    return results



def process_pending_articles(items: list[dict], output_dir="data"):
    """
    Streaming pipeline: Vừa cào, vừa summarize, vừa lưu DB từng bài.
    
    Luồng xử lý cho MỖI bài:
      1. Đánh dấu status = processing (tránh xử lý trùng)
      2. Scrape nội dung (Module 2)
      3. AI Summarize (Module 3) — có VRAM protection
      4. Lưu vào MongoDB (Module 4)
      5. Lưu backup vào JSONL file (theo ngày)
      6. Cập nhật status done/failed
    
    → Nếu pipeline crash giữa chừng, bài đã xử lý vẫn an toàn trong DB.
    """
    from datetime import date
    from module1_fetcher import update_status
    from module3_summarizer import summarize_single
    from module4_storage import save_articles

    # Tạo thư mục + file backup theo ngày
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"raw_articles_{date.today().isoformat()}.jsonl")

    total = len(items)
    success_count = 0
    fail_count = 0

    logger.info(f"🚀 BẮT ĐẦU STREAMING PIPELINE: {total} bài báo")

    # Mở file JSONL ở chế độ append (backup, không ghi đè)
    with open(output_file, 'a', encoding='utf-8') as f:

        for i, item in enumerate(tqdm(items, desc="Tiến độ"), 1):
            url = item["url"]
            url_hash = item["url_hash"]

            try:
                # ── Bước 0: Đánh dấu đang xử lý (tránh pipeline khác pick lại) ──
                update_status(url_hash, "processing")

                # ── Bước 1: Scrape nội dung ──
                scraped = scrape_article(url)

                if not scraped:
                    logger.warning(f"[{i}/{total}] ❌ Scrape thất bại: {url[:60]}")
                    update_status(url_hash, "failed")
                    fail_count += 1
                    continue

                # Trộn metadata (Module 1) + nội dung (Module 2)
                full_article = {**item, **scraped}

                # ── Bước 2: AI Summarize (có VRAM protection) ──
                full_article = summarize_single(full_article)

                # ── Bước 3: Lưu vào MongoDB ngay ──
                save_articles([full_article])

                # ── Bước 4: Backup ra JSONL file ──
                f.write(json.dumps(full_article, ensure_ascii=False, default=str) + "\n")
                f.flush()  # Flush ngay để không mất data nếu crash

                # ── Bước 5: Cập nhật status ──
                status = "done" if full_article.get("status") == "done" else "failed"
                update_status(url_hash, status)

                if status == "done":
                    success_count += 1
                else:
                    fail_count += 1

                logger.info(
                    f"[{i}/{total}] {'✅' if status == 'done' else '⚠️'} {status.upper()} | "
                    f"{full_article.get('method', '?')} | {url[:60]}"
                )

            except Exception as e:
                logger.error(f"[{i}/{total}] 💥 Pipeline error: {url[:60]} — {e}")
                update_status(url_hash, "failed")
                fail_count += 1

    logger.info(
        f"✅ HOÀN THÀNH: {success_count} thành công, "
        f"{fail_count} thất bại / {total} tổng cộng"
    )