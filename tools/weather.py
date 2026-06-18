"""Weather — 天气查询模块（QWeather 多天预报切片）

核心功能：
1. 调用 QWeather /v7/weather/{days} 接口获取多天天气预报
2. 从预报中切片提取指定日期的数据

工作原理：
- 每次请求获取多天预报（今天 + 未来若干天）
- 多个日期的查询可以复用同一次预报请求，减少 API 调用
- MVP 行程窗口 ≤7 天（router 强制限制）
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Dict, List, Optional, Tuple

from tools.cities import get_city
from tools.qweather import QWeatherError, _request

logger = logging.getLogger(__name__)

# 30 min — QWeather forecast updates roughly hourly.
_WEATHER_TTL_SECONDS = 30 * 60

# QWeather 多天预报窗口。Trip window is ≤7 days.
_FORECAST_DAYS = "7d"

# Cache key: (city_id, date) — avoids re-slicing the same day out of a forecast blob.
_weather_cache: Dict[Tuple[str, str], Tuple[Dict, float]] = {}
_weather_lock = threading.Lock()

# 城市名 → QWeather LocationID 的解析缓存（进程级），避免重复 geo 查询。
_location_id_cache: Dict[str, str] = {}
_location_lock = threading.Lock()


def _is_location_id_or_coords(value: str) -> bool:
    """判断字符串是否已经是 QWeather 可直接使用的 location。

    QWeather /v7/weather 的 location 仅接受：
      - 数字 LocationID（如 "101280101"）
      - "lng,lat" 经纬度坐标（如 "113.27,23.13"）
    城市名（如 "guangzhou"、"广州"）会被接口判为 400 Invalid Parameter。
    """
    stripped = value.replace(",", "").replace(".", "").replace("-", "")
    return stripped.isdigit() and stripped != ""


def _resolve_location(city: str) -> Optional[str]:
    """把 agent 传入的城市标识解析成 QWeather 可用的 location。

    - 已经是 LocationID / 经纬度坐标 → 原样返回；
    - 是城市名（拼音/中文/英文）→ 通过 /geo/v2/city/lookup 解析为 LocationID（带缓存）；
    - 解析失败 → 返回 None，由上层按"工具失败"处理。
    """
    if not city:
        return None
    c = city.strip()
    if _is_location_id_or_coords(c):
        return c

    key = c.lower()
    with _location_lock:
        cached = _location_id_cache.get(key)
    if cached:
        return cached

    try:
        info = get_city(c)
    except QWeatherError as e:
        logger.warning("天气定位解析失败 city=%s：%s", city, e)
        return None
    loc_id = info.get("id")
    if not loc_id:
        logger.warning("天气定位解析无结果 city=%s", city)
        return None
    with _location_lock:
        _location_id_cache[key] = loc_id
    logger.info("天气定位解析成功 city=%s -> LocationID=%s（%s）",
                city, loc_id, info.get("name"))
    return loc_id


# ---------------------------------------------------------------------------
# QWeather 真实 API 调用 (Real API)
# ---------------------------------------------------------------------------
def _fetch_forecast(city_id: str) -> List[Dict]:
    """调用 QWeather /v7/weather/{days} 获取多天天气预报

    Args:
        city_id: QWeather 城市 ID

    Returns:
        List[Dict]: 多天的每日预报数据列表
    """
    logger.info("QWeather 预报请求 location=%s", city_id)
    data = _request(
        f"/v7/weather/{_FORECAST_DAYS}",
        params={"location": city_id, "lang": "zh"},
        cache_ttl=_WEATHER_TTL_SECONDS,
    )
    dailies = data.get("daily", [])
    logger.info("QWeather 预报获取成功 location=%s，天数=%d", city_id, len(dailies))
    return dailies


def _text_to_condition(text_day: str) -> str:
    """将 QWeather 中文天气描述映射到我们的 condition 枚举

    Args:
        text_day: QWeather 返回的 textDay 字段（如 "多云"、"小雨"）

    Returns:
        str: "sunny" | "partly_cloudy" | "cloudy" | "rainy"
    """
    if not text_day:
        return "cloudy"
    # 优先匹配雨、雷等降水天气
    if "雨" in text_day or "雷" in text_day:
        return "rainy"
    # 雪天归为 rainy 类（同样建议室内活动）
    if "雪" in text_day:
        return "rainy"
    if "阴" in text_day:
        return "cloudy"
    # 晴+多云 = 多云间晴
    if "晴" in text_day and "多云" in text_day:
        return "partly_cloudy"
    if "晴" in text_day:
        return "sunny"
    if "多云" in text_day or "云" in text_day:
        return "partly_cloudy"
    # 雾霾沙尘等归为阴天
    if "雾" in text_day or "霾" in text_day or "沙" in text_day:
        return "cloudy"
    return "cloudy"


def _daily_to_weatherinfo(daily: Dict) -> Dict:
    """将 QWeather 单日预报数据转换为我们的天气信息格式

    Args:
        daily: QWeather 返回的单日数据字典

    Returns:
        Dict: {condition, temp_high, temp_low, description}
    """
    text_day = daily.get("textDay", "")
    return {
        "condition": _text_to_condition(text_day),
        "temp_high": int(daily.get("tempMax", 0) or 0),
        "temp_low": int(daily.get("tempMin", 0) or 0),
        "description": text_day or "暂无天气数据",
    }


def _real_weather(city_id: str, date: str) -> Optional[Dict]:
    """从 QWeather 真实 API 获取天气

    任何失败（网络错误、业务错误、日期不在预报窗口内）返回 None。

    Args:
        city_id: QWeather 城市 ID
        date: 目标日期 (YYYY-MM-DD)

    Returns:
        Optional[Dict]: 成功返回天气信息，失败返回 None
    """
    # 关键修复：先把城市名解析成 QWeather LocationID，否则接口返回 400。
    location = _resolve_location(city_id)
    if not location:
        logger.warning("天气：无法将 city=%s 解析为 QWeather LocationID，跳过", city_id)
        return None
    try:
        dailies = _fetch_forecast(location)
    except QWeatherError as e:
        logger.warning("QWeather 预报获取失败 city=%s（location=%s）：%s",
                       city_id, location, e)
        return None
    for d in dailies:
        if d.get("fxDate") == date:
            return _daily_to_weatherinfo(d)
    logger.warning("QWeather 预报未包含 date=%s city=%s", date, city_id)
    return None


# ---------------------------------------------------------------------------
# 公共入口函数 (Public API)
# ---------------------------------------------------------------------------
def get_weather(city: str, date: str) -> Optional[Dict]:
    """获取指定城市指定日期的天气信息

    这是模块的公共入口点，会自动处理：
    1. 内存缓存（30 分钟 TTL）
    2. 调用 QWeather 真实 API

    Args:
        city: QWeather 城市 ID（如 "101010100" 北京）
               与 City.id 字段保持一致
        date: 日期字符串（如 "2024-01-15"）

    Returns:
        Optional[Dict]: 天气信息，获取失败返回 None
            - condition: "sunny" | "partly_cloudy" | "cloudy" | "rainy"
            - temp_high: 最高温度
            - temp_low: 最低温度
            - description: 中文描述
    """
    cache_key = (city, date)
    with _weather_lock:
        entry = _weather_cache.get(cache_key)
        # 缓存未过期直接返回
        if entry is not None and time.monotonic() < entry[1]:
            logger.info("天气信息缓存命中 city=%s date=%s", city, date)
            return entry[0]

    result = _real_weather(city, date)

    # 仅缓存成功结果，失败时下次重试
    if result is not None:
        with _weather_lock:
            _weather_cache[cache_key] = (result, time.monotonic() + _WEATHER_TTL_SECONDS)
        logger.info("天气信息缓存成功 city=%s date=%s", city, date)
    return result
