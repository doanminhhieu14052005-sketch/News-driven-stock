"""
Tiện ích dùng chung cho pipeline.

parse_published_at: chuẩn hóa giá trị published_at (scrape từ CafeF) về datetime.

Lý do tồn tại: CafeF trả 2 format ngày khác nhau tùy nơi lấy
  - Trang timelinelist  -> ISO  "2026-05-01T17:39:00"
  - Trang chủ (page 1)  -> slash "01/05/2026 - 09:02"  (DD/MM/YYYY giờ VN)
Nếu lưu thẳng string thô, lúc parse bằng pandas các chuỗi slash sẽ bị
hiểu nhầm theo MM/DD (kiểu Mỹ) -> đảo ngày/tháng hoặc thành NaT.
Vì vậy luôn parse về datetime (giờ VN, naive) NGAY KHI GHI vào DB.
"""

from datetime import datetime
from typing import Optional, Union

# Các format CafeF có thể xuất hiện. Slash luôn là DD/MM/YYYY (giờ VN).
_FORMATS = (
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
    "%d/%m/%Y - %H:%M",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
)


def parse_published_at(value: Union[str, datetime, None]) -> Optional[datetime]:
    """Trả về datetime (giờ VN, naive) hoặc None nếu không parse được.

    - Nếu value đã là datetime -> trả nguyên (idempotent, an toàn khi gọi lại).
    - Hỗ trợ cả ISO lẫn DD/MM/YYYY. DD/MM/YYYY luôn được hiểu day-first.
    """
    if isinstance(value, datetime):
        return value
    if not value:
        return None
    s = str(value).strip()
    if not s:
        return None
    for fmt in _FORMATS:
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    # Phương án cuối: ISO 8601 có offset/seconds lẻ
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
