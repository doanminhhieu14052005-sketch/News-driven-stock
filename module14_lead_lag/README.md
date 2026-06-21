# Module 14 - Lead-Lag Cross-Correlation (RQ3)

Tính tương quan chéo có độ trễ (lead-lag cross-correlation) giữa **sentiment ngành** và
**return ngành** để trả lời RQ3: sentiment của ngành nào *dẫn trước* (lead) return của ngành nào,
và với độ trễ bao nhiêu ngày.

## Input

`data/processed/merged_sentiment_return_wide.csv` (encoding `utf-8-sig`)

- Cột `trade_date`
- 10 cột `sent_<sector>` và 10 cột `ret_<sector>` cho các ngành chuẩn:
  Banking, RealEstate, SteelMaterials, Technology, ConsumerStaples,
  ConsumerDiscretionary, Energy, IndustrialLogistics, Healthcare, Utilities.

## Xử lý

1. Sắp xếp theo `trade_date` tăng dần, ép kiểu numeric.
2. Điền sentiment thiếu = `0` (neutral) — tắt bằng cờ `--no-fill-missing`.
3. Với mỗi cặp `(sent_i, ret_j)` và mỗi độ trễ `k = 0..max_lag` (mặc định `max_lag=5`):
   - Căn `sent_i[t]` với `ret_j.shift(-k)` (tức `ret_j(t+k)`), drop NaN.
   - Tính Pearson correlation bằng `scipy.stats.pearsonr` -> `(corr, pvalue)`, `n_obs` = số điểm dùng.
   - `k > 0` và `corr` có ý nghĩa => sentiment ngành `i` **dẫn trước** return ngành `j` `k` ngày.
4. `best_lag` mỗi cặp = lag có `|corr|` lớn nhất (chỉ xét lag có `n_obs >= 30`).

## Output (`data/processed/`, `utf-8-sig`)

- `lead_lag_all.csv` — `s_sector, r_sector, lag, corr, pvalue, significant, n_obs`
  (10 x 10 x (max_lag+1) = 600 dòng khi max_lag=5).
- `lead_lag_best.csv` — mỗi cặp 1 dòng best lag:
  `s_sector, r_sector, best_lag, corr, pvalue, significant, n_obs` (100 dòng).

`significant` = `pvalue < 0.05`.

## Chạy

```bash
C:/Users/Admin/anaconda3/envs/tf-gpu/python.exe module14_lead_lag/lead_lag_analysis.py
```

### Tham số

| Cờ | Mặc định | Ý nghĩa |
|----|----------|---------|
| `--input` | `data/processed/merged_sentiment_return_wide.csv` | File đầu vào |
| `--all-output` | `data/processed/lead_lag_all.csv` | File kết quả toàn bộ lag |
| `--best-output` | `data/processed/lead_lag_best.csv` | File best lag mỗi cặp |
| `--max-lag` | `5` | Độ trễ tối đa |
| `--no-fill-missing` | (tắt) | KHÔNG điền sentiment thiếu = 0 |

`print_summary` in top 10 quan hệ **lead** (lag>=1) có ý nghĩa thống kê, sắp theo `|corr|` giảm dần.
