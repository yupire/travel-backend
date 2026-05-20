import hashlib
import random
from typing import Dict

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


def get_weather(city: str, date: str) -> Dict:
    """Mock: deterministic weather based on city + date hash."""
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
