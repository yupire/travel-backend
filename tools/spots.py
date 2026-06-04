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


def geocode_spots(city: str, spot_names: List[str]) -> List[Dict[str, Any]]:
    """Map / geocode lookup for a list of spot names within a city.

    Returns [{name, lat, lng, id, is_indoor}] for each name that resolves.
    Names not found in the city's spot database are skipped (not raised) so
    that free-form input from upstream tools / LLM does not blow up planning.
    Real API: swap the body for a 高德/Google Maps geocode call.
    """
    spot_map = get_spot_map(city)
    results: List[Dict[str, Any]] = []
    for name in spot_names:
        match = next(
            (s for s in spot_map.values() if s["name"] == name or s.get("name_en") == name),
            None,
        )
        if not match:
            continue
        results.append({
            "id": match["id"],
            "name": match["name"],
            "lat": match["lat"],
            "lng": match["lng"],
            "is_indoor": match.get("type") == "indoor",
        })
    return results
