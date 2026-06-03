import json
from datetime import datetime, timedelta
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import JsonOutputParser

from models import TripRequest, TripResponse, DayPlan, SpotPlan, WeatherInfo, FoodRec, TransportInfo
from tools.spots import get_spot_map
from tools.weather import get_weather
from tools.foods import get_top_foods, get_nearby_foods
from tools.routes import get_popular_routes
from tools.routing import reorder_by_weather, add_transport_info
from agent.prompts import PLAN_REASONING_PROMPT

_llm = ChatAnthropic(model="claude-haiku-4-5-20251001", max_tokens=2048)

# LCEL chain: prompt → LLM → JSON parser
_reasoning_chain = PLAN_REASONING_PROMPT | _llm | JsonOutputParser()


def _parse_dates(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]


def _build_days_json(raw_days: list) -> str:
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


def _fallback_reasoning(city: str, raw_days: list) -> dict:
    return {
        "days": [{"day": d["day"], "reasoning": "精彩行程，期待您的探索！"} for d in raw_days],
        "summary": f"这是一次精彩的{city}之旅，祝您旅途愉快！",
    }


def plan_trip(request: TripRequest) -> TripResponse:
    dates = _parse_dates(request.start_date, request.end_date)
    total_days = len(dates)

    spot_map = get_spot_map(request.city)
    routes = get_popular_routes(request.city, total_days)
    all_foods = get_top_foods(request.city)

    # Build raw day data with weather-ordered spots, transport, and nearby foods
    raw_days = []
    for i, (date, route_ids) in enumerate(zip(dates, routes)):
        weather_dict = get_weather(request.city, date)
        day_spots = [spot_map[sid] for sid in route_ids if sid in spot_map]
        day_spots = reorder_by_weather(day_spots, weather_dict["condition"])
        day_spots = add_transport_info(day_spots)
        for spot in day_spots:
            spot["nearby_foods"] = get_nearby_foods(spot, all_foods, limit=4)
        raw_days.append({"day": i + 1, "date": date, "weather": weather_dict, "spots": day_spots})

    # Generate per-day reasoning and trip summary via LCEL chain
    try:
        reasoning_data = _reasoning_chain.invoke({
            "city": request.city,
            "start_date": request.start_date,
            "end_date": request.end_date,
            "days_json": _build_days_json(raw_days),
        })
    except Exception:
        reasoning_data = _fallback_reasoning(request.city, raw_days)

    reasoning_map = {d["day"]: d["reasoning"] for d in reasoning_data.get("days", [])}

    itinerary = []
    for d in raw_days:
        spots = []
        for s in d["spots"]:
            transport = TransportInfo(**s["transport_from_prev"]) if s.get("transport_from_prev") else None
            foods = [FoodRec(**f) for f in s.get("nearby_foods", [])]
            spots.append(SpotPlan(
                id=s["id"], name=s["name"], lat=s["lat"], lng=s["lng"],
                duration_min=s["duration_min"], open_time=s["open_time"],
                ticket=s["ticket"], type=s["type"], tags=s["tags"],
                description=s["description"], nearby_foods=foods,
                transport_from_prev=transport,
            ))
        itinerary.append(DayPlan(
            day=d["day"], date=d["date"],
            weather=WeatherInfo(**d["weather"]),
            spots=spots,
            reasoning=reasoning_map.get(d["day"], ""),
        ))

    return TripResponse(
        city=request.city,
        start_date=request.start_date,
        end_date=request.end_date,
        total_days=total_days,
        itinerary=itinerary,
        summary=reasoning_data.get("summary", ""),
    )
