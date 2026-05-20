import json
import os
from typing import List, Dict

_DATA_PATH = os.path.join(os.path.dirname(__file__), "../data/mock_data.json")

with open(_DATA_PATH, encoding="utf-8") as f:
    _DATA = json.load(f)


def get_cities() -> List[Dict]:
    """Mock: returns list of supported cities."""
    return _DATA["cities"]


def get_city_ids() -> List[str]:
    return [c["id"] for c in _DATA["cities"]]
