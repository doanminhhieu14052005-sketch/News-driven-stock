"""
Module 3: AI Summarizer & Extractor
- Gọi LLM (Ollama local hoặc Gemini) với structured prompt
- Validate output bằng Pydantic
- Retry tối đa 3 lần nếu JSON lỗi
"""

import gc
import json
import logging
import re
import time
from typing import Any, Literal, Optional

import httpx
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from config import (
    LLM_BACKEND, LLM_MAX_RETRIES,
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    GEMINI_API_KEY, GROQ_API_KEYS, GROQ_MODEL,
    MAX_INPUT_CHARS_OLLAMA, MAX_INPUT_CHARS_GEMINI, MAX_INPUT_CHARS_GROQ,
    VRAM_COOLDOWN_SECONDS, ENABLE_GC_CLEANUP,
)

logger = logging.getLogger(__name__)
_groq_key_index = 0


# ── Pydantic Schema ───────────────────────────────────────────────

class ArticleSummary(BaseModel):
    summary: list[str]          # 3 gạch đầu dòng
    tickers: list[str]          # ["FPT", "HPG"] hoặc []
    impact: Literal["Positive", "Negative", "Neutral"]
    sentiment_score: Optional[float] = None  # điểm liên tục [-1, +1] cho Granger/lead-lag
    key_metrics: dict[str, Any] # {"Doanh thu": "1000 tỷ"} hoặc {}
    sector: Optional[str] = None  # "Ngân hàng", "Bất động sản"...

    @field_validator("tickers")
    @classmethod
    def uppercase_tickers(cls, v):
        return [t.upper().strip() for t in v if t.strip()]

    @field_validator("sentiment_score", mode="before")
    @classmethod
    def coerce_score(cls, v):
        """Ép về float trong [-1, 1]. Trả None nếu không parse được (sẽ suy ra từ impact)."""
        if v is None or v == "":
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        return max(-1.0, min(1.0, f))

    @model_validator(mode="after")
    def fill_score_from_impact(self):
        """Nếu LLM quên trả sentiment_score, suy ra giá trị mặc định từ impact
        để chuỗi sentiment luôn có số liên tục và nhất quán dấu với impact."""
        if self.sentiment_score is None:
            self.sentiment_score = {"Positive": 0.5, "Negative": -0.5, "Neutral": 0.0}[self.impact]
        return self

    @field_validator("key_metrics", mode="before")
    @classmethod
    def coerce_metrics(cls, v):
        """Ép tất cả value trong key_metrics về str để tránh crash khi LLM trả số."""
        if isinstance(v, dict):
            return {str(k): str(val) for k, val in v.items()}
        return v or {}

    @field_validator("summary", mode="before")
    @classmethod
    def process_summary(cls, v):
        if isinstance(v, str):
            v = [v]
        if isinstance(v, list):
            v = [str(x).strip() for x in v if str(x).strip()]
            if not v:
                return ["Không có thông tin tóm tắt."]
            return v[:5] # Lấy tối đa 5 điểm nếu AI lỡ viết quá dài
        return v


# ── Prompt ───────────────────────────────────────────────────────

SYSTEM_PROMPT = """Bạn là chuyên gia phân tích tài chính Việt Nam.
Nhiệm vụ: Đọc bài báo tài chính và trả về JSON THUẦN TÚY (không có markdown, không có ```).

Schema bắt buộc:
{
  "summary": ["điểm 1", "điểm 2", "điểm 3"],
  "tickers": ["MÃ1", "MÃ2"],
  "impact": "Positive" | "Negative" | "Neutral",
  "sentiment_score": 0.0,
  "key_metrics": {"Chỉ số": "Giá trị"},
  "sector": "Một ngành trong danh sách, hoặc null"
}

Quy tắc:
- summary: 2–3 gạch đầu dòng, mỗi điểm dưới 25 từ
- tickers: HỈ chứa các mã chứng khoán HOSE/HNX (viết hoa). NẾU BÀI BÁO KHÔNG ĐỀ CẬP ĐẾN MÃ CHỨNG KHOÁN NÀO, BẮT BUỘC TRẢ VỀ: "tickers": []. TUYỆT ĐỐI KHÔNG trả về null.
- impact: đánh giá tác động đến thị trường/doanh nghiệp
- sentiment_score: SỐ THỰC liên tục trong [-1.0, +1.0] thể hiện mức độ tích cực/tiêu cực của tin với thị trường. -1.0 = cực kỳ tiêu cực, 0.0 = trung tính, +1.0 = cực kỳ tích cực. Phải CÙNG DẤU với impact (Positive > 0, Negative < 0, Neutral ≈ 0) và phản ánh CƯỜNG ĐỘ (tin sốc mạnh gần ±1.0, tin nhẹ gần ±0.2).
- key_metrics: chỉ trích xuất nếu bài có số liệu cụ thể
- sector: Phân loại bài vào MỘT ngành, dùng CHÍNH XÁC một trong các nhãn tiếng Việt sau:
    "Ngân hàng" | "Bất động sản" | "Thép & Vật liệu" | "Công nghệ & Viễn thông" |
    "Hàng tiêu dùng thiết yếu" | "Bán lẻ & Tiêu dùng" | "Dầu khí" |
    "Công nghiệp & Logistics" | "Y tế & Dược" | "Tiện ích & Điện"
  Quy tắc chọn:
    • Nếu bài nói về một doanh nghiệp/ngành cụ thể (kể cả khi nhắc mã cổ phiếu) -> BẮT BUỘC chọn ngành gần nhất.
    • Gợi ý: thép/tôn/vật liệu xây dựng/hóa chất/cao su -> "Thép & Vật liệu"; cảng/hàng không/vận tải/logistics -> "Công nghiệp & Logistics"; ô tô/vàng bạc/bán lẻ -> "Bán lẻ & Tiêu dùng"; thực phẩm/đồ uống/sữa/nông nghiệp -> "Hàng tiêu dùng thiết yếu"; điện/nước/năng lượng tái tạo -> "Tiện ích & Điện".
    • CHỈ trả "sector": null khi tin THUẦN vĩ mô/toàn thị trường (VN-Index, lãi suất, tỷ giá, chính sách chung) không gắn ngành cụ thể.
- KHÔNG giải thích thêm, chỉ JSON"""

USER_TEMPLATE = """Bài báo:
---
{text}
---
JSON:"""


# ── LLM Backends ─────────────────────────────────────────────────

def _call_ollama(text: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text[:MAX_INPUT_CHARS_OLLAMA])},
        ],
        "stream": False,
        "format": "json",   # ép Ollama trả JSON hợp lệ -> ít lỗi parse, đỡ retry
        "options": {"temperature": 0.1},
    }
    with httpx.Client(timeout=120) as client:
        resp = client.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload)
        resp.raise_for_status()
        return resp.json()["message"]["content"]


def _call_gemini(text: str) -> str:
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    )
    payload = {
        "contents": [{"parts": [{"text": (
            SYSTEM_PROMPT + "\n\n" +
            USER_TEMPLATE.format(text=text[:MAX_INPUT_CHARS_GEMINI])
        )}]}],
        "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
    }
    with httpx.Client(timeout=30) as client:
        resp = client.post(url, json=payload)
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def _call_groq(text: str) -> str:
    url = "https://api.groq.com/openai/v1/chat/completions"
    current_key = GROQ_API_KEYS[_groq_key_index % len(GROQ_API_KEYS)] if GROQ_API_KEYS else ""
    headers = {
        "Authorization": f"Bearer {current_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(text=text[:MAX_INPUT_CHARS_GROQ])}
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"} # Bật JSON mode của Groq
    }
    with httpx.Client(timeout=90) as client:
        resp = client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]


def _extract_json(raw: str) -> str:
    """
    Trích xuất JSON object đầu tiên từ output LLM.
    Dùng brace-counting thay vì greedy regex để tránh bắt nhầm
    ký tự } trong text thừa phía sau JSON.
    """
    # Bước 1: Xóa markdown fences (cả mở và đóng)
    raw = re.sub(r"```(?:json)?", "", raw).strip()

    # Bước 2: Tìm JSON object bằng đếm ngoặc {}
    start = raw.find("{")
    if start == -1:
        return raw  # Không tìm thấy { → trả nguyên

    depth = 0
    in_string = False
    escape_next = False

    for i in range(start, len(raw)):
        ch = raw[i]

        if escape_next:
            escape_next = False
            continue

        if ch == "\\":
            if in_string:
                escape_next = True
            continue

        if ch == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return raw[start:i + 1]

    # Fallback: không tìm thấy cặp {} hoàn chỉnh → trả nguyên
    return raw


# ── Main ─────────────────────────────────────────────────────────

def summarize(raw_text: str) -> Optional[ArticleSummary]:
    """
    Gọi LLM, parse + validate JSON.
    Retry tối đa LLM_MAX_RETRIES lần.
    """
    if LLM_BACKEND == "ollama":
        call_fn = _call_ollama
    elif LLM_BACKEND == "gemini":
        call_fn = _call_gemini
    else:
        call_fn = _call_groq

    global _groq_key_index
    keys_tried_this_round = 0
    max_retries = LLM_MAX_RETRIES
    if LLM_BACKEND == "groq" and len(GROQ_API_KEYS) > 1:
        max_retries = max(LLM_MAX_RETRIES, len(GROQ_API_KEYS) * 2)

    for attempt in range(1, max_retries + 1):
        try:
            raw = call_fn(raw_text)
            json_str = _extract_json(raw)
            data = json.loads(json_str)
            result = ArticleSummary.model_validate(data)
            logger.info(f"Summarized OK (attempt {attempt})")
            if LLM_BACKEND == "gemini":
                time.sleep(5) # Giữ nhịp độ API (Limit: 15 RPM)
            return result

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"Parse error attempt {attempt}: {e}")
            if LLM_BACKEND == "gemini":
                time.sleep(5) # Tránh gọi dồn dập khi retry
        except httpx.HTTPError as e:
            error_msg = str(e).lower()
            resp = getattr(e, "response", None)
            status = resp.status_code if resp is not None else None
            body = (resp.text if resp is not None else "").lower()
            logger.error(f"LLM HTTP error attempt {attempt}: {e}")

            # Key Groq bị khoá / không hợp lệ (400 organization_restricted, 401, 403)
            # -> KHÔNG retry cùng key, nhảy sang key khác ngay
            if (LLM_BACKEND == "groq" and len(GROQ_API_KEYS) > 1
                    and status in (400, 401, 403)
                    and ("organization_restricted" in body
                         or "invalid_api_key" in body
                         or "restricted" in body)):
                keys_tried_this_round += 1
                if keys_tried_this_round < len(GROQ_API_KEYS):
                    _groq_key_index += 1
                    logger.warning(
                        f"⛔ Key Groq bị khoá/không hợp lệ. Nhảy sang key thứ "
                        f"{(_groq_key_index % len(GROQ_API_KEYS)) + 1}. Thử lại ngay..."
                    )
                    continue
                else:
                    logger.error("💀 TẤT CẢ key Groq đều bị khoá/không hợp lệ. Bỏ qua bài này.")
                    break

            # Xử lý 429 Rate Limit bằng Exponential Backoff
            if hasattr(e, "response") and e.response is not None and e.response.status_code == 429:
                if LLM_BACKEND == "groq" and len(GROQ_API_KEYS) > 1:
                    keys_tried_this_round += 1
                    if keys_tried_this_round < len(GROQ_API_KEYS):
                        _groq_key_index += 1
                        logger.warning(f"⚠️ Bị giới hạn 429. Tự động nhảy sang Key Groq thứ {(_groq_key_index % len(GROQ_API_KEYS)) + 1}. Thử lại ngay...")
                        continue
                    else:
                        keys_tried_this_round = 0
                        wait_time = 30 * attempt
                        logger.warning(f"⚠️ TẤT CẢ các Key đều bị 429. Đang nghỉ {wait_time}s...")
                        time.sleep(wait_time)
                        continue
                else:
                    wait_time = 30 * attempt
                    logger.warning(f"⚠️ Bị giới hạn tốc độ API (429). Đang nghỉ {wait_time}s...")
                    time.sleep(wait_time)
                    continue

            # Phát hiện OOM → chờ VRAM giải phóng rồi retry
            if "out of memory" in error_msg or "oom" in error_msg:
                logger.warning(
                    f"⚠️ VRAM OOM detected! Waiting {VRAM_COOLDOWN_SECONDS}s..."
                )
                time.sleep(VRAM_COOLDOWN_SECONDS)
                gc.collect()

    logger.error("Summarizer failed after max retries")
    return None


def _cleanup_memory():
    """Dọn dẹp memory sau mỗi lần summarize."""
    if ENABLE_GC_CLEANUP:
        gc.collect()
        # Nếu có torch (dùng GPU trực tiếp), xóa cache CUDA
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass  # Không có torch → bỏ qua (Ollama tự quản lý VRAM)


def summarize_single(article: dict) -> dict:
    """
    Summarize 1 bài + cleanup VRAM sau khi xong.
    Dùng cho streaming pipeline (scrape → summarize → save từng bài).
    """
    summary = summarize(article["raw_text"])
    if summary:
        article["summary_json"] = summary.model_dump()
        article["status"] = "done"
    else:
        article["summary_json"] = None
        article["status"] = "failed"

    _cleanup_memory()
    return article


def summarize_batch(articles: list[dict]) -> list[dict]:
    """Thêm summary_json vào mỗi article dict (batch mode, backward compat)."""
    results = []
    for article in articles:
        article = summarize_single(article)
        results.append(article)
    return results