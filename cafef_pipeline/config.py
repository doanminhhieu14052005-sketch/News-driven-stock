import os
from dotenv import load_dotenv

load_dotenv()

# === CafeF ===
# api_id = zoneid của chuyên mục (dùng cho timelinelist/{api_id}/{page}.chn).
# LƯU Ý: id 18831 thực chất là chuyên mục CHỨNG KHOÁN, không phải Doanh nghiệp.
# Trước đây key "doanh_nghiep" trỏ nhầm vào 18831 -> dữ liệu cào ra phần lớn là tin chứng khoán.
CAFEF_CATEGORIES = {
    "vi_mo": {
        "url": "https://cafef.vn/vi-mo-dau-tu.chn",
        "api_id": "18833"
    },
    "chung_khoan": {                                            # đổi tên: 18831 = Chứng khoán (đã cào tới trang 350)
        "url": "https://cafef.vn/thi-truong-chung-khoan.chn",
        "api_id": "18831"
    },
    "doanh_nghiep": {                                           # SỬA: Doanh nghiệp thật sự = 18836
        "url": "https://cafef.vn/doanh-nghiep.chn",
        "api_id": "18836"
    },
    "tai_chinh_nh": {                                           # MỚI: lấp ngành Banking
        "url": "https://cafef.vn/tai-chinh-ngan-hang.chn",
        "api_id": "18834"
    },
    "bat_dong_san": {                                           # MỚI: lấp ngành Real Estate
        "url": "https://cafef.vn/bat-dong-san.chn",
        "api_id": "18835"
    },
    "hang_hoa": {                                               # MỚI: lấp ngành Steel & Energy
        "url": "https://cafef.vn/hang-hoa-nguyen-lieu.chn",
        "api_id": "18839"
    }
}

# Thêm biến cấu hình độ sâu mặc định
SCRAPE_DEPTH = 700  # Trần an toàn (~525 trang = 1 năm + dự phòng). Mốc thời gian bên dưới mới là điều kiện dừng chính.

# === Resumable Crawl ===
# Mục tiêu: cào đủ ~1 năm dữ liệu. Dừng cào 1 danh mục khi gặp bài cũ hơn CRAWL_LOOKBACK_DAYS.
CRAWL_LOOKBACK_DAYS = 365   # Chỉ giữ bài trong vòng N ngày gần nhất (None = bỏ qua, cào theo SCRAPE_DEPTH)
MAX_PAGES_PER_RUN = None    # None = cào hết tới khi chạm mốc 1 năm trong 1 lần chạy. Đặt số (vd 50) nếu muốn chia nhỏ.
CRAWL_PROGRESS_FILE = os.path.join(os.path.dirname(__file__), "data", "crawl_progress.json")

REQUEST_DELAY = (1.5, 3.0)  # seconds, random uniform
REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# === MongoDB ===
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = "cafef_news"
MONGO_COLLECTION = "articles"

# === LLM ===
LLM_BACKEND = os.getenv("LLM_BACKEND", "groq")  # "ollama" | "gemini" | "groq"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEYS_RAW = os.getenv("GROQ_API_KEYS", os.getenv("GROQ_API_KEY", ""))
GROQ_API_KEYS = [k.strip() for k in GROQ_API_KEYS_RAW.split(",") if k.strip()]
GROQ_MODEL = os.getenv("GROQ_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")
LLM_MAX_RETRIES = 3

# === Pipeline ===
# Các khung giờ pipeline tự động chạy (format "HH:MM")
SCHEDULE_TIMES = ["07:00", "12:00", "18:00"]
BATCH_SIZE = 50000  # Đủ rút cạn backlog (~33k pending) trong 1 lượt cày overnight.
                    # Steady-state qua GitHub Actions: có thể hạ về 5000 (job 6h sẽ tự cắt, resumable).

# === VRAM Protection ===
MAX_INPUT_CHARS_OLLAMA = 3000   # Giới hạn chars gửi vào Ollama (tránh tràn VRAM)
MAX_INPUT_CHARS_GEMINI = 12000   # Tăng nhẹ cho Gemini
MAX_INPUT_CHARS_GROQ = 15000     # Groq thường có context window tốt
VRAM_COOLDOWN_SECONDS = 30      # Thời gian chờ nếu gặp OOM error
ENABLE_GC_CLEANUP = True        # Chạy gc.collect() sau mỗi bài summarize