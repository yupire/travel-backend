import math
from typing import List, Dict, Any
from tools.transport import get_transport


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lng = math.radians(lng2 - lng1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def reorder_by_weather(spots: List[Dict], condition: str) -> List[Dict]:
    """Rainy day → indoor spots first; sunny/cloudy → outdoor first."""
    if condition == "rainy":
        return sorted(spots, key=lambda s: (0 if s.get("type") == "indoor" else 1))
    return sorted(spots, key=lambda s: (0 if s.get("type") == "outdoor" else 1))


def add_transport_info(spots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Adds transport_from_prev to each spot based on distance from previous."""
    result = []
    for i, spot in enumerate(spots):
        entry = dict(spot)
        if i > 0:
            prev = spots[i - 1]
            dist = haversine_km(prev["lat"], prev["lng"], spot["lat"], spot["lng"])
            entry["transport_from_prev"] = get_transport(dist)
        else:
            entry["transport_from_prev"] = None
        result.append(entry)
    return result
