"""
Migration 1 lần: chuẩn hóa field `published_at` trong MongoDB về kiểu datetime.

Bối cảnh: trước đây published_at lưu dạng string với 2 format lẫn lộn
(ISO "2026-05-01T17:39:00" và slash "01/05/2026 - 09:02"). Slash là DD/MM/YYYY
giờ VN nhưng dễ bị parse nhầm thành MM/DD -> đảo ngày/tháng hoặc NaT.

Script này parse mọi published_at dạng string về datetime (day-first cho slash)
và ghi đè bằng kiểu BSON date thật. Idempotent: chạy lại không đổi gì.

Cách dùng:
    python migrate_published_at.py            # DRY-RUN: chỉ báo cáo, không ghi
    python migrate_published_at.py --apply    # Thực thi cập nhật
"""

import sys
from collections import Counter

from pymongo import MongoClient, UpdateOne

from config import MONGO_URI, MONGO_DB, MONGO_COLLECTION
from utils import parse_published_at


def main(apply: bool) -> None:
    col = MongoClient(MONGO_URI)[MONGO_DB][MONGO_COLLECTION]

    # Chỉ những doc có published_at là string mới cần convert.
    query = {"published_at": {"$type": "string"}}
    total_str = col.count_documents(query)
    total_all = col.count_documents({})
    print(f"Tổng docs: {total_all} | published_at dạng string: {total_str}")

    ops = []
    stats = Counter()
    failed_samples = []

    for doc in col.find(query, {"published_at": 1}):
        raw = doc["published_at"]
        dt = parse_published_at(raw)
        if dt is None:
            stats["unparseable"] += 1
            if len(failed_samples) < 20:
                failed_samples.append(raw)
            continue
        stats["converted"] += 1
        ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"published_at": dt}}))

    print(f"  -> parse được   : {stats['converted']}")
    print(f"  -> KHÔNG parse  : {stats['unparseable']}")
    if failed_samples:
        print("  Mẫu không parse được:")
        for s in failed_samples:
            print("     ", repr(s))

    if not apply:
        print("\n[DRY-RUN] Chưa ghi gì. Chạy lại với --apply để thực thi.")
        return

    if not ops:
        print("\nKhông có gì để cập nhật.")
        return

    print(f"\nĐang ghi {len(ops)} cập nhật...")
    # Chia batch để tránh payload quá lớn
    BATCH = 1000
    modified = 0
    for i in range(0, len(ops), BATCH):
        res = col.bulk_write(ops[i:i + BATCH], ordered=False)
        modified += res.modified_count
    print(f"Hoàn tất. modified_count = {modified}")

    # Kiểm tra lại
    remaining = col.count_documents({"published_at": {"$type": "string"}})
    print(f"Còn lại published_at dạng string: {remaining}")


if __name__ == "__main__":
    main(apply="--apply" in sys.argv)
