import math
from typing import List, Dict, Any, Optional, Tuple
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


def _greedy_order_from_anchor(
    anchor: Tuple[float, float],
    spots: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Greedy nearest-neighbor ordering starting from `anchor` (lat, lng).

    Picks the spot closest to the current cursor, then advances the cursor
    to that spot's coordinates. Order is stable (preserves input order on
    ties) so identical distances do not shuffle the result.
    """
    remaining = list(spots)
    ordered: List[Dict[str, Any]] = []
    cursor = anchor
    while remaining:
        # min with stable key: (distance, original_index)
        idx, _ = min(
            enumerate(remaining),
            key=lambda pair: (
                haversine_km(cursor[0], cursor[1], pair[1]["lat"], pair[1]["lng"]),
                pair[0],
            ),
        )
        chosen = remaining.pop(idx)
        ordered.append(chosen)
        cursor = (chosen["lat"], chosen["lng"])
    return ordered


def optimize_by_distance_progression(
    routes_per_day: List[List[Dict[str, Any]]],
    day1_anchor: Tuple[float, float],
) -> List[List[Dict[str, Any]]]:
    """Reorder each day's spots so travel progresses outward — no backtracking.

    Day 1's first spot is the closest to `day1_anchor` (typically the
    city-center / hotel area), and each subsequent spot is the closest
    unvisited one to the previous day's terminal spot. This guarantees the
    cumulative path is monotonically expanding rather than revisiting earlier
    areas.

    Falls back to the input order if `routes_per_day` is empty.
    """
    if not routes_per_day:
        return routes_per_day

    optimized: List[List[Dict[str, Any]]] = []
    prev_terminal: Tuple[float, float] = day1_anchor

    for day_index, day_spots in enumerate(routes_per_day):
        if not day_spots:
            optimized.append([])
            continue
        anchor = day1_anchor if day_index == 0 else prev_terminal
        ordered = _greedy_order_from_anchor(anchor, day_spots)
        optimized.append(ordered)
        last = ordered[-1]
        prev_terminal = (last["lat"], last["lng"])

    return optimized
