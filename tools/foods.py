import json
import os
import math
from typing import List, Dict, Any

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_data.json")

with open(_DATA_PATH, encoding="utf-8") as f:
    _FOODS_DB: Dict[str, List[Dict]] = json.load(f)["foods"]


def get_top_foods(city: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Mock: returns top N food recommendations for a city."""
    key = city.lower()
    return _FOODS_DB.get(key, [])[:limit]


def _haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> int:
    """Distance in meters between two coordinates."""
    R = 6371000
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    return int(R * 2 * math.asin(math.sqrt(a)))


def get_nearby_foods(spot: Dict, city_foods: List[Dict], limit: int = 4) -> List[Dict[str, Any]]:
    """Returns N foods closest to the given spot from city_foods list."""
    scored = [
        (f, _haversine_m(spot["lat"], spot["lng"], f["lat"], f["lng"]))
        for f in city_foods
    ]
    scored.sort(key=lambda x: x[1])
    result = []
    for food, dist in scored[:limit]:
        result.append({
            "id": food["id"],
            "name": food["name"],
            "cuisine": food["cuisine"],
            "price_range": food["price_range"],
            "rating": food["rating"],
            "distance_m": dist,
        })
    return result
