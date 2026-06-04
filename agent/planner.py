"""Trip planner — split into 6 step functions per the /plan capability spec.

Each capability from the spec maps to exactly one function below. The
top-level `plan_trip` is a thin orchestrator that wires them together.

Capability → function map:
  1. 按目的地搜索每日天气           → fetch_daily_weather
  2. 查询目的地景点                 → lookup_city_spots
  3. 调用地图查询位置（数组化）     → geocode_and_attach_locations
  4. 设置室内外属性                 → mark_indoor_outdoor
  5. 按距离递进 + 天气分配每日行程  → optimize_by_progression_and_weather
  6. 输出 JSON 行程 + 规划意图       → generate_reasoning + to_response
"""
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

from models import (
    TripRequest,
    TripResponse,
    DayPlan,
    SpotPlan,
    WeatherInfo,
    FoodRec,
    TransportInfo,
)
from tools.spots import get_spot_map, geocode_spots
from tools.weather import get_weather
from tools.foods import get_top_foods, get_nearby_foods
from tools.routes import get_popular_routes
from tools.routing import (
    reorder_by_weather,
    add_transport_info,
    optimize_by_distance_progression,
)
from agent.prompts import PLAN_REASONING_PROMPT

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)

# LCEL chain: prompt → LLM → JSON parser
_reasoning_chain = PLAN_REASONING_PROMPT | _llm | JsonOutputParser()


# ---------------------------------------------------------------------------
# Capability 1: 天气 — fetch daily weather for each date in the trip window.
# ---------------------------------------------------------------------------
def fetch_daily_weather(city: str, dates: List[str]) -> List[Dict[str, Any]]:
    """Look up deterministic weather for (city, date) for every day of the trip."""
    return [get_weather(city, date) for date in dates]


# ---------------------------------------------------------------------------
# Capability 2: 景点 — look up the candidate attraction pool for the city.
# ---------------------------------------------------------------------------
def lookup_city_spots(city: str) -> Dict[str, Dict[str, Any]]:
    """Return the spot database for the city as {spot_id: spot_dict}."""
    return get_spot_map(city)


# ---------------------------------------------------------------------------
# Capability 3: 地图查询位置 — resolve names to coordinates, recorded as an
# array of {id, name, lat, lng, is_indoor} for downstream distance math.
# ---------------------------------------------------------------------------
def geocode_and_attach_locations(
    city: str,
    spot_dicts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach a geocoded location array entry to each spot dict.

    The spot dicts from `lookup_city_spots` already carry lat/lng (mock data);
    this step makes the "map query → array" pipeline explicit, so swapping in
    a real geocoding API (高德 / Google Maps) is a one-function change.
    """
    names = [s["name"] for s in spot_dicts]
    locations = geocode_spots(city, names)
    by_id = {loc["id"]: loc for loc in locations}
    for spot in spot_dicts:
        loc = by_id.get(spot["id"])
        if loc:
            spot["location"] = loc
    return spot_dicts


# ---------------------------------------------------------------------------
# Capability 4: 室内/外属性 — normalize the indoor/outdoor flag.
# Reads from data when present; falls back to True for type=="indoor".
# Real API integration: the upstream spot record should set is_indoor directly.
# ---------------------------------------------------------------------------
def mark_indoor_outdoor(spots: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Add explicit `is_indoor: bool` to each spot (data-driven, not inferred)."""
    for spot in spots:
        # Prefer explicit field if the data source already set it
        if "is_indoor" not in spot:
            spot["is_indoor"] = spot.get("type") == "indoor"
    return spots


# ---------------------------------------------------------------------------
# Capability 5: 距离递进 + 天气分配 — two-stage optimization.
# Stage A: greedy nearest-neighbor from city anchor on day 1, then from the
#          previous day's terminal spot on subsequent days. Guarantees the
#          trip "moves outward" — no backtracking to earlier areas.
# Stage B: within each day, weather-aware reordering so outdoor spots lead
#          on good-weather days and indoor spots lead on rainy days.
# ---------------------------------------------------------------------------
def optimize_by_progression_and_weather(
    routes_per_day: List[List[Dict[str, Any]]],
    weather_per_day: List[Dict[str, Any]],
    day1_anchor: Tuple[float, float],
) -> List[List[Dict[str, Any]]]:
    """Stage A: distance-progression ordering. Stage B: weather reordering."""
    progressed = optimize_by_distance_progression(routes_per_day, day1_anchor)
    return [
        reorder_by_weather(day, weather_per_day[i]["condition"])
        for i, day in enumerate(progressed)
    ]


# ---------------------------------------------------------------------------
# Capability 6: 推理意图 — generate per-day reasoning + trip summary via LLM.
# Falls back to a generic message if the LLM call fails.
# ---------------------------------------------------------------------------
def _build_days_summary_json(raw_days: List[Dict[str, Any]]) -> str:
    summary = [
        {
            "day": d["day"],
            "date": d["date"],
            "weather": d["weather"]["condition"],
            "temp": f"{d['weather']['temp_low']}-{d['weather']['temp_high']}°C",
            "spots": [s["name"] for s in d["spots"]],
            "foods": [f["name"] for s in d["spots"] for f in s.get("nearby_foods", [])][:5],
        }
        for d in raw_days
    ]
    return json.dumps(summary, ensure_ascii=False, indent=2)


def _fallback_reasoning(city: str, raw_days: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "days": [{"day": d["day"], "reasoning": "精彩行程，期待您的探索！"} for d in raw_days],
        "summary": f"这是一次精彩的{city}之旅，祝您旅途愉快！",
    }


def generate_reasoning(
    city: str,
    start_date: str,
    end_date: str,
    raw_days: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Per-day reasoning + trip summary. LCEL chain with graceful fallback."""
    try:
        return _reasoning_chain.invoke({
            "city": city,
            "start_date": start_date,
            "end_date": end_date,
            "days_json": _build_days_summary_json(raw_days),
        })
    except Exception:
        return _fallback_reasoning(city, raw_days)


# ---------------------------------------------------------------------------
# Helpers used by the orchestrator.
# ---------------------------------------------------------------------------
def _parse_dates(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]


# Rough city-center anchors for day-1 starting point. Real implementation
# would take the hotel location as the anchor from the user request.
_CITY_ANCHORS: Dict[str, Tuple[float, float]] = {
    "singapore": (1.2839, 103.8514),   # Raffles Place
    "tokyo":     (35.6812, 139.7671),   # Tokyo Station
    "paris":     (48.8606, 2.3376),     # Louvre
    "beijing":   (39.9087, 116.3975),   # Tiananmen
    "bangkok":   (13.7563, 100.5018),   # Grand Palace
}


def _attach_nearby_foods(
    day_spots: List[Dict[str, Any]],
    all_foods: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    for spot in day_spots:
        spot["nearby_foods"] = get_nearby_foods(spot, all_foods, limit=4)
    return day_spots


def _build_spot_plan(spot: Dict[str, Any]) -> SpotPlan:
    transport = (
        TransportInfo(**spot["transport_from_prev"])
        if spot.get("transport_from_prev")
        else None
    )
    foods = [FoodRec(**f) for f in spot.get("nearby_foods", [])]
    return SpotPlan(
        id=spot["id"],
        name=spot["name"],
        lat=spot["lat"],
        lng=spot["lng"],
        duration_min=spot["duration_min"],
        open_time=spot["open_time"],
        ticket=spot["ticket"],
        type=spot["type"],
        is_indoor=spot.get("is_indoor", spot.get("type") == "indoor"),
        tags=spot["tags"],
        description=spot["description"],
        nearby_foods=foods,
        transport_from_prev=transport,
    )


def to_response(
    city: str,
    start_date: str,
    end_date: str,
    total_days: int,
    raw_days: List[Dict[str, Any]],
    reasoning_data: Dict[str, Any],
) -> TripResponse:
    """Convert internal raw_days → public TripResponse (schema adapter)."""
    reasoning_map = {d["day"]: d["reasoning"] for d in reasoning_data.get("days", [])}
    itinerary: List[DayPlan] = []
    for d in raw_days:
        spots = [_build_spot_plan(s) for s in d["spots"]]
        itinerary.append(
            DayPlan(
                day=d["day"],
                date=d["date"],
                weather=WeatherInfo(**d["weather"]),
                spots=spots,
                reasoning=reasoning_map.get(d["day"], ""),
                is_indoor_outdoor_filter=d.get("is_indoor_outdoor_filter", False),
            )
        )
    return TripResponse(
        city=city,
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
        itinerary=itinerary,
        summary=reasoning_data.get("summary", ""),
    )


# ---------------------------------------------------------------------------
# Top-level orchestrator — wires the 6 capability steps together.
# ---------------------------------------------------------------------------
def plan_trip(request: TripRequest) -> TripResponse:
    dates = _parse_dates(request.start_date, request.end_date)
    total_days = len(dates)

    # Step 1: daily weather
    weather_per_day = fetch_daily_weather(request.city, dates)

    # Step 2: spot pool for the city
    spot_map = lookup_city_spots(request.city)

    # Step 3: geocode + record as location array on each spot
    geocode_and_attach_locations(request.city, list(spot_map.values()))

    # Step 4: mark indoor/outdoor on every spot
    mark_indoor_outdoor(list(spot_map.values()))

    # Pull the route skeletons (spot ID lists per day)
    routes_ids = get_popular_routes(request.city, total_days)
    all_foods = get_top_foods(request.city)

    # Resolve route IDs to spot dicts for the progression stage
    raw_routes: List[List[Dict[str, Any]]] = [
        [spot_map[sid] for sid in day_ids if sid in spot_map]
        for day_ids in routes_ids
    ]

    # Step 5: distance progression + weather reordering
    anchor = _CITY_ANCHORS.get(request.city.lower(), list(spot_map.values())[0] and (list(spot_map.values())[0]["lat"], list(spot_map.values())[0]["lng"]))
    optimized_routes = optimize_by_progression_and_weather(
        raw_routes, weather_per_day, anchor,
    )

    # Build raw_days: add transport, nearby foods, and the
    # indoor/outdoor-filter flag (True if any reorder was driven by rain).
    raw_days: List[Dict[str, Any]] = []
    for i, (date, day_spots, weather) in enumerate(
        zip(dates, optimized_routes, weather_per_day)
    ):
        day_spots = add_transport_info(day_spots)
        day_spots = _attach_nearby_foods(day_spots, all_foods)
        raw_days.append({
            "day": i + 1,
            "date": date,
            "weather": weather,
            "spots": day_spots,
            "is_indoor_outdoor_filter": weather["condition"] == "rainy",
        })

    # Step 6: LLM reasoning + summary, then build the response
    reasoning_data = generate_reasoning(
        request.city, request.start_date, request.end_date, raw_days,
    )

    return to_response(
        request.city,
        request.start_date,
        request.end_date,
        total_days,
        raw_days,
        reasoning_data,
    )
