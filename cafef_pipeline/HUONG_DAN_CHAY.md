# Hướng dẫn chạy CafeF Pipeline

Pipeline thu thập + xử lý tin tức tài chính CafeF cho đề tài *News-driven Sector
Sentiment Spillover Network (HOSE)*. Mỗi bài được cào → tóm tắt + chấm
`sentiment_score` bằng LLM → lưu MongoDB.

> **Cày backlog gấp?** Nhảy xuống [§4. Cày backlog nhiều máy](#4-cày-backlog-nhiều-máy-song-song).

---

## 1. Yêu cầu

| Thành phần | Ghi chú |
|---|---|
| Python 3.10+ | Khuyến nghị conda env `tf-gpu` (đã có sẵn các thư viện) |
| MongoDB Atlas | Dùng chung 1 cluster cho cả nhóm (chuỗi `MONGO_URI`) |
| **Ollama** + GPU | Backend LLM local. Cần GPU ≥ 6GB VRAM cho `qwen2.5:7b` |

---

## 2. Cài đặt

```bash
# 1) Lấy code
git clone https://github.com/doanminhhieu14052005-sketch/News-driven-stock.git
cd News-driven-stock/cafef_pipeline

# 2) Cài thư viện Python
pip install -r requirements.txt

# 3) Cài Ollama (https://ollama.com/download) rồi kéo model
ollama pull qwen2.5:7b
```

Đảm bảo Ollama đang chạy (app khay hệ thống trên Windows, hoặc `ollama serve`).
Kiểm tra nhanh: `curl http://localhost:11434/api/tags`.

---

## 3. Cấu hình `.env`

Tạo file `.env` trong thư mục `cafef_pipeline` (file này đã được `.gitignore`,
**không commit**):

```ini
# Backend LLM: "ollama" (local, khuyến nghị) | "groq" | "gemini"
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# Bắt buộc: chuỗi kết nối MongoDB Atlas dùng chung của nhóm
MONGO_URI=mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority

# Tùy chọn (chỉ cần nếu đổi LLM_BACKEND sang groq/gemini)
# GROQ_API_KEY=key1,key2,...
# GEMINI_API_KEY=...
```

> ⚠️ **Nhất quán model:** Cả nhóm nên dùng **cùng `OLLAMA_MODEL=qwen2.5:7b`**.
> Nếu trộn nhiều model, `sentiment_score` sẽ không đồng nhất → phải ghi vào
> phần Limitations của báo cáo.

---

## 4. Cày backlog nhiều máy song song

Để xử lý hết các bài `pending` đang tồn trong DB. **Chạy được trên nhiều máy
cùng lúc** — mỗi bài được claim nguyên tử (`find_one_and_update`) nên không máy
nào trùng việc, không tốn tài nguyên gấp đôi.

Trên **mỗi máy** (sau khi cài đặt + cấu hình `.env` như trên):

```bash
conda activate tf-gpu      # hoặc env Python của bạn
cd News-driven-stock/cafef_pipeline
python drain_backlog.py
```

- Script **không cào URL mới**, chỉ rút cạn hàng đợi sẵn có.
- Máy nào chết giữa chừng → bài `processing` treo sẽ được máy khác nhận lại sau
  15 phút (không kẹt vĩnh viễn).
- Dừng an toàn bất cứ lúc nào bằng **Ctrl+C** (bài đã xong vẫn nằm trong DB).

Tốc độ tham khảo: ~250–360 bài/giờ/máy với `qwen2.5:7b` trên RTX 4060.
3 máy ≈ 1 ngày cho ~33k bài.

---

## 5. Thu tin định kỳ (steady-state)

Sau khi đã hết backlog, chạy orchestrator để **cào tin mới** + xử lý theo lịch
(`SCHEDULE_TIMES` trong `config.py`). Chỉ cần **1 máy** chạy việc này:

```bash
python orchestrator.py
```

Hoặc dùng GitHub Actions (`.github/workflows/pipeline.yml`) chạy tự động 3 lần/ngày
(cần đặt secrets `MONGO_URI`, `GROQ_API_KEY` trong repo settings).

---

## 6. Theo dõi tiến độ

Đếm nhanh trạng thái trong MongoDB:

```python
from pymongo import MongoClient
from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION
col = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLLECTION]
for s in ["pending", "processing", "done", "failed"]:
    print(s, col.count_documents({"status": s}))
# Số bài đã có điểm sentiment (run mới):
print("with_score", col.count_documents({"summary_json.sentiment_score": {"$exists": True}}))
```

---

## 7. Tiện ích một lần

`migrate_published_at.py` — chuẩn hóa field `published_at` về kiểu datetime
(đã chạy 2026-06). Dùng lại nếu DB lẫn lại format string:

```bash
python migrate_published_at.py            # dry-run, chỉ báo cáo
python migrate_published_at.py --apply    # thực thi
```

---

## 8. Cấu trúc module

| File | Vai trò |
|---|---|
| `config.py` | Cấu hình (danh mục CafeF, MongoDB, LLM, lịch chạy) |
| `module1_fetcher.py` | Cào danh sách URL + dedup + `claim_next_pending()` (chia việc đa máy) |
| `module2_scraper.py` | Tải & bóc nội dung sạch từng bài |
| `module3_summarizer.py` | Gọi LLM → tóm tắt + `impact` + `sentiment_score` + sector (Pydantic validate) |
| `module4_storage.py` | Kết nối MongoDB, lưu bài, tạo index |
| `module5_vnstock_price.py` | Lấy giá cổ phiếu theo ngành (vnstock) |
| `orchestrator.py` | Điều phối cào + xử lý theo lịch |
| `drain_backlog.py` | Rút cạn backlog đa máy (process-only) |
| `utils.py` | Helper dùng chung (`parse_published_at`) |

---

## 9. Xử lý sự cố

| Triệu chứng | Cách khắc phục |
|---|---|
| `could not connect to Ollama` | Bật Ollama (`ollama serve` / app khay). Kiểm tra `OLLAMA_BASE_URL`. |
| Model chưa có | `ollama pull qwen2.5:7b` |
| Ollama OOM / chậm | GPU thiếu VRAM. Đóng app khác, hoặc dùng model nhỏ hơn (lưu ý §3 về nhất quán). |
| Bài kẹt `processing` | Tự nhận lại sau 15 phút. Hoặc reset thủ công: đặt `status="pending"`. |
| `conda run` lỗi `charmap`/cp1252 | Đừng chạy script in tiếng Việt qua `conda run`; hãy `conda activate` rồi chạy `python ...` trực tiếp. |
