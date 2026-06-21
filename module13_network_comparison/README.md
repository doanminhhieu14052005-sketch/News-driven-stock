# Module 13 — Network Comparison (RQ2)

So sánh **mạng tin tức** (sentiment spillover, từ Granger module 10) với **mạng giá**
(price correlation, từ module 12) trên cùng 10 ngành HOSE.

## Mục đích (RQ2)
Cấu trúc mạng lan truyền sentiment có tương đồng với cấu trúc tương quan giá không?

## Input
- `data/processed/granger_results.csv` — kết quả Granger sentiment→sentiment (module 10, mode `sent2sent`).
- `data/processed/price_network_adjacency.csv` — ma trận kề mạng giá (module 12).

## Xử lý
- Dựng ma trận kề **có hướng** của mạng tin tức từ cột significance (mặc định `significant_raw_05`).
- **Đối xứng hóa** mạng tin tức (cạnh i–j nếu i→j hoặc j→i) để so với mạng giá vô hướng.
- Tính trên các ô off-diagonal: Frobenius distance/similarity, Jaccard, tương quan
  Pearson & Spearman, và **QAP/Mantel permutation test** (hoán vị nhãn node 2000 lần, seed=42).

## Output
- `data/processed/network_comparison.csv` — bảng metric (metric, value).

## Cách chạy
```
python module13_network_comparison/compare_networks.py
# tuỳ chọn dùng cạnh đã hiệu chỉnh FDR:
python module13_network_comparison/compare_networks.py --news-significance-column significant_fdr_05
```

## Diễn giải
- QAP p < 0.05 → hai mạng tương đồng có ý nghĩa (tin tức phản ánh cấu trúc giá).
- QAP p ≥ 0.05 → hai mạng khác nhau (news và price mã hóa chiều thông tin riêng) — vẫn là finding có giá trị.
