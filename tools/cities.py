"""City list — backed by QWeather /geo/v2/city/lookup.

The cold-start city list is assembled by calling the lookup endpoint for a
small set of seed keywords, then deduping by the QWeather city ``id``. Results
are cached in-process for 24h; the cache survives between requests but is
cleared on process restart.

On QWeather failure (missing key, network error, etc.) we raise QWeatherError
— the cities endpoint surfaces this as HTTP 503 to the client, because cold
start must be real data (no mock fallback here).
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List

from tools.qweather import QWeatherError, _request, is_configured

log = logging.getLogger(__name__)

# 24h — the city list is stable data.
_CITIES_TTL_SECONDS = 24 * 60 * 60

# Seed queries for the cold-start city list. Hand-picked to cover the cities
# the rest of the system knows about (spots/foods/routes keyed on lowercase
# slugs). Lookup returns many matches per query; we dedupe by id.
_SEED_QUERIES = [
    "北京", "上海", "广州", "深圳", "成都", "杭州", "西安", "重庆", "厦门", "青岛",
    "香港", "澳门", "台北",
    "东京", "大阪", "京都", "札幌", "奈良", "横滨",
    "首尔", "釜山",
    "曼谷", "清迈", "普吉", "新加坡",
    "巴黎", "伦敦", "罗马", "米兰", "巴塞罗那", "马德里", "柏林", "阿姆斯特丹",
    "纽约", "洛杉矶", "旧金山", "芝加哥", "华盛顿",
    "悉尼", "墨尔本", "奥克兰",
    "迪拜", "伊斯坦布尔",
]

_lock = threading.Lock()
_cities_cache: List[Dict] | None = None
_cities_cache_expires_at: float = 0.0


def _fetch_all_cities() -> List[Dict]:
    """Call /geo/v2/city/lookup for each seed and merge deduped results."""
    seen: Dict[str, Dict] = {}
    for q in _SEED_QUERIES:
        try:
            data = _request(
                "/geo/v2/city/lookup",
                params={"location": q, "range": "cn" if any("一" <= ch <= "鿿" for ch in q) else "world", "number": 20, "lang": "zh"},
                cache_ttl=_CITIES_TTL_SECONDS,
            )
        except QWeatherError as e:
            log.warning("QWeather lookup %r 失败：%s", q, e)
            continue
        for item in data.get("location", []):
            cid = item.get("id")
            if not cid or cid in seen:
                continue
            # Skip non-city entries (e.g. scenic spots that show up in lookup).
            if item.get("type") not in ("city", "scenic"):
                continue
            seen[cid] = {
                "id": cid,
                "name": item.get("name", ""),
                "name_en": item.get("name", ""),  # QWeather /lookup returns Chinese; reuse
                "country": item.get("country", ""),
                "lat": float(item.get("lat", 0.0) or 0.0),
                "lng": float(item.get("lon", 0.0) or 0.0),
                "adm1": item.get("adm1", ""),
                "adm2": item.get("adm2", ""),
                "fx_link": item.get("fxLink", ""),
            }
    if not seen:
        raise QWeatherError(
            "城市列表为空：QWeather 全部 seed 查询均失败，请检查 .env 凭据或网络"
        )
    # Stable order: country then name. Frontend shows the user's locale, so
    # grouping by country makes the dropdown scan-friendly.
    return sorted(seen.values(), key=lambda c: (c["country"], c["name"], c["id"]))


def get_cities() -> List[Dict]:
    """Return the cached city list. Cold-start triggers a real QWeather fetch."""
    global _cities_cache, _cities_cache_expires_at
    import time

    if not is_configured():
        raise QWeatherError(
            "QWeather 未配置 QWEATHER_PRIVATE_KEY。请在 backend/.env 设置 Ed25519 私钥。"
        )
    with _lock:
        if _cities_cache is not None and time.monotonic() < _cities_cache_expires_at:
            return _cities_cache
        log.info("城市列表缓存未命中，调用 QWeather /geo/v2/city/lookup 重建（%d 个 seed）", len(_SEED_QUERIES))
        _cities_cache = _fetch_all_cities()
        _cities_cache_expires_at = time.monotonic() + _CITIES_TTL_SECONDS
        return _cities_cache


def get_city_ids() -> List[str]:
    return [c["id"] for c in get_cities()]
