# Module 6 - Session Bucket Mapper

## Muc tieu

Module 6 anh xa thoi diem xuat ban tin tuc `published_at` sang phien giao
dich va ngay giao dich phu hop de tranh data leakage khi ghep tin tuc voi
sector return.

Module them cac cot:

- `published_at_vn`
- `session_bucket`
- `trade_date`
- `mapped_reason`
- `is_mapped`

## Vi tri trong du an

Module 6 la module doc lap cua du an lon **News-driven Sector Sentiment
Spillover Network**. Module nay khong phai module con cua VNStock. VNStock
Module 5 chi cung cap trading calendar thong qua output sector return.

## Input

Mac dinh:

- Bai viet: `data/raw/articles.csv`
- Trading calendar Module 5: `VNStock/data/processed/daily_sector_price_wide.csv`

File bai viet phai co cot `published_at`. Tat ca cot input khac duoc giu lai.
Trading calendar phai co cot `trade_date`.

## Output

Mac dinh:

`data/processed/articles_session_mapped.csv`

Output giu tat ca cot goc va them nam cot mapping cua Module 6.
CSV output duoc ghi bang `utf-8-sig` de ho tro tieng Viet khi mo bang Excel,
GitHub preview, va VS Code.

## Quy tac bucket

Timezone mac dinh la `Asia/Ho_Chi_Minh`.

| Thoi diem tren ngay giao dich | Bucket | Trade date |
| --- | --- | --- |
| 00:00 den truoc 09:00 | `PRE_MARKET` | Ngay hien tai |
| 09:00 den truoc 11:30 | `MORNING` | Ngay hien tai |
| 11:30 den truoc 14:45 | `AFTERNOON` | Ngay hien tai |
| Tu 14:45 tro di | `NEXT_MORNING` | Ngay giao dich ke tiep |
| Ngay khong giao dich | `NEXT_MORNING` | Ngay giao dich ke tiep |

Module chi dung danh sach ngay trong trading calendar. Module khong tu suy
doan cuoi tuan, ngay le, hoac ngay nghi.

Neu `published_at` khong hop le, bucket la `UNKNOWN` va `is_mapped=False`.
Neu calendar khong con ngay giao dich ke tiep, `trade_date` de trong,
`is_mapped=False`, va module log warning.

## Cach chay

Chay voi duong dan mac dinh:

```bash
python module6_session_bucket/session_mapper.py
```

Truyen duong dan tuy chinh:

```bash
python module6_session_bucket/session_mapper.py --articles-input data/raw/articles.csv --trading-calendar VNStock/data/processed/daily_sector_price_wide.csv --output data/processed/articles_session_mapped.csv
```

## Export CafeF tu MongoDB

Module 6 co script export input tu MongoDB:

```bash
python module6_session_bucket/export_cafef_from_mongo.py --output data/raw/articles.csv
```

Script doc `MONGO_URI`, `MONGO_DB_NAME`, va `MONGO_COLLECTION` tu bien moi
truong hoac file `.env` local. Khong commit `.env` hoac `data/raw/articles.csv`.
