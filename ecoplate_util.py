"""Small shared helpers.

Time zone: Render servers run in UTC, but a canteen in India lives in IST (UTC+05:30). Using the server's
`date.today()` would reject records for "today" during the first hours of the Indian morning (the server still
thinks it is yesterday) and would mis-order forecasts. EcoPlate therefore uses ECOPLATE_TIMEZONE (default
Asia/Kolkata) for "today" and "now".
"""
from __future__ import annotations

import os
from datetime import date, datetime, timedelta, timezone


def _tz():
    name = (os.environ.get("ECOPLATE_TIMEZONE") or "Asia/Kolkata").strip()
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(name)
    except Exception:                       # tzdata missing on a slim image, or a bad name
        if name in ("Asia/Kolkata", "Asia/Calcutta", "IST"):
            return timezone(timedelta(hours=5, minutes=30), "IST")
        return timezone.utc


def local_now():
    return datetime.now(_tz())


def local_today() -> date:
    return local_now().date()
