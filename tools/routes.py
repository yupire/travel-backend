import json
import os
from typing import List

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_data.json")

with open(_DATA_PATH, encoding="utf-8") as f:
    _ROUTES_DB = json.load(f)["routes"]


def get_popular_routes(city: str, days: int) -> List[List[str]]:
    """Mock: returns popular routes (spot IDs per day) for city × N days."""
    city_routes = _ROUTES_DB.get(city.lower(), {})
    max_days = max(int(k) for k in city_routes) if city_routes else 1

    # Clamp to max defined days
    key = str(min(days, max_days))
    base_routes = city_routes.get(key, city_routes.get("3", [[]]))[:]

    # If requested days exceed defined routes, pad with shortened days
    while len(base_routes) < days:
        base_routes.append(base_routes[len(base_routes) % len(base_routes)])

    return base_routes[:days]
