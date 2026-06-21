"""
Kiểm tra nhanh tiến độ cày backlog: bao nhiêu máy đang chạy, đã xong bao nhiêu bài,
tốc độ và ước tính thời gian còn lại.

Cách chạy (trong thư mục cafef_pipeline, env đã activate):
    python check_progress.py
"""

import sys
import time
from datetime import datetime, timezone

from pymongo import MongoClient

from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION

# In tiếng Việt không lỗi trên Windows
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ACTIVE_WINDOW_SEC = 120   # bài 'processing' claim trong N giây gần đây = máy còn sống
SAMPLE_SEC = 15           # đo tốc độ trong bao nhiêu giây


def main():
    col = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLLECTION]
    now = datetime.now(timezone.utc)

    done = col.count_documents({"status": "done"})
    pending = col.count_documents({"status": "pending"})
    processing = col.count_documents({"status": "processing"})
    failed = col.count_documents({"status": "failed"})
    with_score = col.count_documents({"summary_json.sentiment_score": {"$exists": True}})

    # Đếm máy đang chạy = số bài 'processing' có claimed_at mới
    active = 0
    for d in col.find({"status": "processing"}, {"claimed_at": 1, "_id": 0}):
        ca = d.get("claimed_at")
        if ca is None:
            continue
        if ca.tzinfo is None:
            ca = ca.replace(tzinfo=timezone.utc)
        if (now - ca).total_seconds() < ACTIVE_WINDOW_SEC:
            active += 1

    # Đo tốc độ
    print("Đang đo tốc độ trong %ds..." % SAMPLE_SEC)
    d1 = col.count_documents({"summary_json.sentiment_score": {"$exists": True}})
    time.sleep(SAMPLE_SEC)
    d2 = col.count_documents({"summary_json.sentiment_score": {"$exists": True}})
    rate_hr = (d2 - d1) / SAMPLE_SEC * 3600

    print("=" * 44)
    print(" TIẾN ĐỘ CÀY BACKLOG")
    print("=" * 44)
    print(" Máy đang chạy (active)   : %d" % active)
    print(" Đã xong run mới (score)  : %d" % with_score)
    print(" Done tổng                : %d" % done)
    print(" Còn pending              : %d" % pending)
    print(" Đang xử lý / Failed      : %d / %d" % (processing, failed))
    print("-" * 44)
    print(" Tốc độ                   : ~%.0f bài/giờ" % rate_hr)
    if rate_hr > 0:
        hrs = pending / rate_hr
        print(" Ước tính còn lại         : ~%.1f giờ (~%.1f ngày)" % (hrs, hrs / 24))
    else:
        print(" Tốc độ ~0 — kiểm tra xem có máy nào đang chạy không")
    print("=" * 44)


if __name__ == "__main__":
    main()
