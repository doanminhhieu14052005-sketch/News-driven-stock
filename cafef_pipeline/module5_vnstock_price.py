"""
Module 5: vnstock Price Store
- Lấy dữ liệu OHLCV daily cho 10 ngành × 3 proxy (theo DOCX Section 2.3)
- Tính equal-weighted sector return
- Output: data/raw_stock_prices.csv + data/sector_returns.csv
"""

import os
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import time as _time

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("module5_vnstock_price")

# ── 10 Ngành GICS-adapted × 3 Proxy (theo DOCX Section 2.3) ────────
SECTORS = {
    "Banking":                  ["VCB", "BID", "CTG"],
    "RealEstate":               ["VIC", "VHM", "NVL"],
    "Steel_Materials":          ["HPG", "HSG", "NKG"],
    "Technology":               ["FPT", "CMG", "VGI"],
    "ConsumerStaples":          ["MSN", "SAB", "VNM"],
    "ConsumerDiscretionary":    ["MWG", "PNJ", "DGW"],
    "Energy_OilGas":            ["GAS", "PLX", "PVD"],
    "Industrial_Logistics":     ["GMD", "HAH", "ACV"],
    "Healthcare_Pharma":        ["DHG", "IMP", "DMC"],
    "Utilities_Power":          ["POW", "NT2", "PPC"],
}

START_DATE = "2025-05-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
RAW_OUTPUT = "data/raw_stock_prices.csv"
SECTOR_RETURN_OUTPUT = "data/sector_returns.csv"


def fetch_all_prices() -> pd.DataFrame:
    """Lấy OHLCV cho tất cả 30 mã cổ phiếu, dùng vnstock 4.x API."""
    from vnstock.api.quote import Quote

    os.makedirs("data", exist_ok=True)
    all_frames = []
    request_count = 0
    BATCH_LIMIT = 18  # vnstock free tier: 20 requests/phút, dùng 18 để an toàn

    for sector, tickers in SECTORS.items():
        logger.info(f"── Ngành: {sector} ({', '.join(tickers)}) ──")
        for ticker in tickers:
            # Rate-limit: chờ 65s sau mỗi 18 requests
            if request_count > 0 and request_count % BATCH_LIMIT == 0:
                logger.info(f"⏳ Đã gửi {request_count} requests, chờ 65s để tránh rate-limit...")
                _time.sleep(65)

            try:
                q = Quote(symbol=ticker, source="VCI")
                df = q.history(start=START_DATE, end=END_DATE)
                request_count += 1

                if df is None or df.empty:
                    logger.warning(f"  {ticker}: Không có dữ liệu")
                    continue

                df["ticker"] = ticker
                df["sector"] = sector
                all_frames.append(df)
                logger.info(f"  {ticker}: {len(df)} dòng OK")

                _time.sleep(0.5)  # Delay nhỏ giữa các request

            except Exception as e:
                logger.error(f"  {ticker}: LỖI — {e}")
                request_count += 1

    if not all_frames:
        logger.critical("Không lấy được bất kỳ dữ liệu nào!")
        return pd.DataFrame()

    raw = pd.concat(all_frames, ignore_index=True)
    raw = raw.sort_values(by=["sector", "ticker", "time"]).reset_index(drop=True)

    # Tính price_return từng mã
    raw["price_return"] = raw.groupby("ticker")["close"].pct_change()

    # Lưu raw
    raw.to_csv(RAW_OUTPUT, index=False)
    logger.info(f"Đã lưu raw OHLCV: {RAW_OUTPUT} ({len(raw)} dòng)")

    return raw


def compute_sector_returns(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Tính equal-weighted average return cho mỗi ngành mỗi ngày.
    Output: DataFrame index=date, columns=10 sectors.
    Đây là input chính cho M3 (Granger) và M4 (Lead-lag).
    """
    if raw.empty:
        return pd.DataFrame()

    # Pivot: mỗi ngày, mỗi ngành → trung bình return của 3 proxy
    sector_ret = raw.pivot_table(
        index="time",
        columns="sector",
        values="price_return",
        aggfunc="mean"
    )

    # Sắp xếp theo thời gian
    sector_ret = sector_ret.sort_index()

    # Fill NaN dòng đầu tiên = 0 (chưa có return)
    sector_ret = sector_ret.fillna(0)

    # Lưu
    sector_ret.to_csv(SECTOR_RETURN_OUTPUT)
    logger.info(f"Đã lưu sector returns: {SECTOR_RETURN_OUTPUT} ({sector_ret.shape})")

    return sector_ret


if __name__ == "__main__":
    raw = fetch_all_prices()
    if not raw.empty:
        sector_ret = compute_sector_returns(raw)
        print("\n── Sector Return (5 dòng cuối) ──")
        print(sector_ret.tail())
        print(f"\nTổng: {sector_ret.shape[0]} ngày × {sector_ret.shape[1]} ngành")
