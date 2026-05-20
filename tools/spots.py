import json
import os
from typing import List, Dict, Any

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_data.json")

with open(_DATA_PATH, encoding="utf-8") as f:
    _SPOTS_DB: Dict[str, List[Dict]] = json.load(f)["spots"]


def get_top_spots(city: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Mock: returns top N tourist spots for a city."""
    key = city.lower()
    spots = _SPOTS_DB.get(key, [])
    return spots[:limit]


def get_spot_map(city: str) -> Dict[str, Dict]:
    """Returns {spot_id: spot_dict} for a city."""
    return {s["id"]: s for s in get_top_spots(city)}
