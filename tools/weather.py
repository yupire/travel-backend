"""Weather — QWeather /v7/weather/3d with hash-based mock fallback.

Real implementation: one GET per (city_id, 3-day-window) returns a 3-day
forecast, then we slice the day we need by ``fxDate``. Per-day calls are
sliced from a single 3d fetch to amortize — the trip window in MVP is ≤7
days (router enforces this), so we batch up to 3 dates per request.

On any QWeather failure (missing key, network error, code != 200, date not in
returned window) we fall back to the deterministic mock that ships in this
file. Mock is also the default for historical dates (the 3d forecast endpoint
only returns today + 2 future days).
"""
from __future__ import annotations

import hashlib
import logging
import random
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from tools.qweather import QWeatherError, _request, is_configured

log = logging.getLogger(__name__)

# 30 min — QWeather 3d forecast updates roughly hourly.
_WEATHER_TTL_SECONDS = 30 * 60

# QWeather 3d endpoint supports today + 2 future days. Trip window is ≤7 days.
# To keep the implementation simple we use a single 3d fetch and fall back to
# mock for any date outside the returned window (rare — would only happen if
# the user picks a date > 2 days from today).
_FORECAST_DAYS = "3d"

# Cache key: (city_id, date) — avoids re-slicing the same day out of a 3d blob.
_weather_cache: Dict[Tuple[str, str], Tuple[Dict, float]] = {}
_weather_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Mock (deterministic) fallback — same algorithm as the original mock.
# ---------------------------------------------------------------------------
_CITY_BASE_TEMPS = {
    "singapore": {"high": 32, "low": 26},
    "tokyo":     {"high": 22, "low": 14},
    "paris":     {"high": 20, "low": 12},
    "beijing":   {"high": 25, "low": 15},
    "bangkok":   {"high": 34, "low": 27},
}

_CONDITIONS = ["sunny", "partly_cloudy", "cloudy", "rainy"]
_WEIGHTS    = [0.45,   0.30,          0.15,    0.10]
_TEMP_OFFSETS = {"sunny": 2, "partly_cloudy": 0, "cloudy": -1, "rainy": -3}
_DESCRIPTIONS = {
    "sunny":        "晴空万里，非常适合户外活动",
    "partly_cloudy": "多云间晴，天气舒适宜人",
    "cloudy":       "阴天多云，建议携带外套",
    "rainy":        "有雨天气，建议优先安排室内景点",
}


def _mock_weather(city: str, date: str) -> Dict:
    """Mock: deterministic weather based on city + date hash.

    The ``city`` arg here is the QWeather city id (e.g. "101010100") or any
    string — we lowercase it but don't require it to match the legacy slugs.
    """
    seed = int(hashlib.md5(f"{city.lower()}{date}".encode()).hexdigest()[:8], 16)
    random.seed(seed)
    condition = random.choices(_CONDITIONS, weights=_WEIGHTS)[0]
    base = _CITY_BASE_TEMPS.get(city.lower(), {"high": 25, "low": 18})
    offset = _TEMP_OFFSETS[condition]
    return {
        "condition": condition,
        "temp_high": base["high"] + offset,
        "temp_low":  base["low"]  + offset,
        "description": _DESCRIPTIONS[condition],
    }


# ---------------------------------------------------------------------------
# QWeather real API
# ---------------------------------------------------------------------------
def _fetch_forecast(city_id: str) -> List[Dict]:
    """GET /v7/weather/3d — returns the 3 daily entries from the API."""
    data = _request(
        f"/v7/weather/{_FORECAST_DAYS}",
        params={"location": city_id, "lang": "zh"},
        cache_ttl=_WEATHER_TTL_SECONDS,
    )
    return data.get("daily", [])


def _text_to_condition(text_day: str) -> str:
    """Map a QWeather Chinese textDay (e.g. '多云', '小雨') to our condition enum."""
    if not text_day:
        return "cloudy"
    if "雨" in text_day or "雷" in text_day:
        return "rainy"
    if "雪" in text_day:
        return "rainy"  # collapse snow into rain-like indoor-priority path
    if "阴" in text_day:
        return "cloudy"
    if "晴" in text_day and "多云" in text_day:
        return "partly_cloudy"
    if "晴" in text_day:
        return "sunny"
    if "多云" in text_day or "云" in text_day:
        return "partly_cloudy"
    if "雾" in text_day or "霾" in text_day or "沙" in text_day:
        return "cloudy"
    return "cloudy"


def _daily_to_weatherinfo(daily: Dict) -> Dict:
    text_day = daily.get("textDay", "")
    return {
        "condition": _text_to_condition(text_day),
        "temp_high": int(daily.get("tempMax", 0) or 0),
        "temp_low": int(daily.get("tempMin", 0) or 0),
        "description": text_day or "暂无天气数据",
    }


def _real_weather(city_id: str, date: str) -> Optional[Dict]:
    """Try QWeather real API. Return None on any failure (caller falls back)."""
    try:
        dailies = _fetch_forecast(city_id)
    except QWeatherError as e:
        log.warning("QWeather 3d 获取失败 city=%s：%s — fallback to mock", city_id, e)
        return None
    for d in dailies:
        if d.get("fxDate") == date:
            return _daily_to_weatherinfo(d)
    log.warning("QWeather 3d 未包含 date=%s city=%s — fallback to mock", date, city_id)
    return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def get_weather(city: str, date: str) -> Dict:
    """Return weather for (city, date).

    ``city`` is a QWeather city id (e.g. "101010100") — same as the value
    stored in ``City.id`` after the cities.py rewrite. Falls back to the
    deterministic mock when QWeather is unconfigured or the date is outside
    the 3-day forecast window.
    """
    cache_key = (city, date)
    with _weather_lock:
        entry = _weather_cache.get(cache_key)
        if entry is not None and time.monotonic() < entry[1]:
            return entry[0]

    if is_configured():
        result = _real_weather(city, date) or _mock_weather(city, date)
    else:
        result = _mock_weather(city, date)

    with _weather_lock:
        _weather_cache[cache_key] = (result, time.monotonic() + _WEATHER_TTL_SECONDS)
    return result
