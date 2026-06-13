"""Weather — 天气查询模块（QWeather 3天预报 + 确定性 mock 降级）

核心功能：
1. 调用 QWeather /v7/weather/3d 接口获取 3 天天气预报
2. 从 3 天预报中切片提取指定日期的数据
3. 任何 API 失败时降级到基于哈希的确定性 mock

工作原理：
- 每次请求获取 3 天预报（今天 + 未来 2 天）
- 多个日期的查询可以复用同一次 3d 请求，减少 API 调用
- MVP 行程窗口 ≤7 天（router 强制限制）
- 超出 3 天预报窗口的日期自动使用 mock

降级策略：
- 私钥未配置 → mock
- 网络错误 → mock
- 业务错误 (code != 200) → mock
- 日期不在返回的 3 天窗口内 → mock
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
# 确定性 Mock 降级方案 (Deterministic Mock Fallback)
# ---------------------------------------------------------------------------
_CITY_BASE_TEMPS = {
    "singapore": {"high": 32, "low": 26},
    "tokyo":     {"high": 22, "low": 14},
    "paris":     {"high": 20, "low": 12},
    "beijing":   {"high": 25, "low": 15},
    "bangkok":   {"high": 34, "low": 27},
}

_CONDITIONS = ["sunny", "partly_cloudy", "cloudy", "rainy"]
_WEIGHTS    = [0.45,   0.30,          0.15,    0.10]  # 天气状况概率权重
_TEMP_OFFSETS = {"sunny": 2, "partly_cloudy": 0, "cloudy": -1, "rainy": -3}  # 不同天气的温度修正
_DESCRIPTIONS = {
    "sunny":        "晴空万里，非常适合户外活动",
    "partly_cloudy": "多云间晴，天气舒适宜人",
    "cloudy":       "阴天多云，建议携带外套",
    "rainy":        "有雨天气，建议优先安排室内景点",
}


def _mock_weather(city: str, date: str) -> Dict:
    """生成基于城市+日期哈希的确定性 mock 天气

    使用 MD5(city + date) 作为随机种子，确保同一城市同一日期
    始终返回相同的天气数据，方便测试和开发。

    Args:
        city: QWeather 城市 ID（如 "101010100"）或任意字符串
        date: 日期字符串（如 "2024-01-15"）

    Returns:
        Dict: 包含 condition, temp_high, temp_low, description 的天气信息
    """
    # 使用城市+日期的哈希值作为随机种子，确保结果确定性
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
# QWeather 真实 API 调用 (Real API)
# ---------------------------------------------------------------------------
def _fetch_forecast(city_id: str) -> List[Dict]:
    """调用 QWeather /v7/weather/3d 获取 3 天天气预报

    Args:
        city_id: QWeather 城市 ID

    Returns:
        List[Dict]: 3 天的每日预报数据列表
    """
    data = _request(
        f"/v7/weather/{_FORECAST_DAYS}",
        params={"location": city_id, "lang": "zh"},
        cache_ttl=_WEATHER_TTL_SECONDS,
    )
    return data.get("daily", [])


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
    """尝试从 QWeather 真实 API 获取天气

    任何失败（网络错误、业务错误、日期不在 3 天窗口内）返回 None，
    由调用方降级到 mock。

    Args:
        city_id: QWeather 城市 ID
        date: 目标日期 (YYYY-MM-DD)

    Returns:
        Optional[Dict]: 成功返回天气信息，失败返回 None
    """
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
# 公共入口函数 (Public API)
# ---------------------------------------------------------------------------
def get_weather(city: str, date: str) -> Dict:
    """获取指定城市指定日期的天气信息

    这是模块的公共入口点，会自动处理：
    1. 内存缓存（30 分钟 TTL）
    2. 优先使用 QWeather 真实 API
    3. 任何失败自动降级到 mock

    Args:
        city: QWeather 城市 ID（如 "101010100" 北京）
               与 City.id 字段保持一致
        date: 日期字符串（如 "2024-01-15"）

    Returns:
        Dict: 天气信息
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
            return entry[0]

    # 优先尝试真实 API，失败则降级到 mock
    if is_configured():
        result = _real_weather(city, date) or _mock_weather(city, date)
    else:
        result = _mock_weather(city, date)

    # 写入缓存
    with _weather_lock:
        _weather_cache[cache_key] = (result, time.monotonic() + _WEATHER_TTL_SECONDS)
    return result
