import json
import anthropic
from datetime import datetime, timedelta
from typing import List

from models import TripRequest, TripResponse, DayPlan, SpotPlan, WeatherInfo, FoodRec, TransportInfo
from tools.spots import get_top_spots, get_spot_map
from tools.weather import get_weather
from tools.foods import get_top_foods, get_nearby_foods
from tools.routes import get_popular_routes
from tools.routing import reorder_by_weather, add_transport_info

client = anthropic.Anthropic()


def _parse_dates(start: str, end: str) -> List[str]:
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]


def _generate_reasoning(city: str, start_date: str, end_date: str, days_data: list) -> dict:
    """Call Claude once to generate per-day reasoning and trip summary."""
    days_summary = []
    for d in days_data:
        days_summary.append({
            "day": d["day"],
            "date": d["date"],
            "weather": d["weather"]["condition"],
            "temp": f"{d['weather']['temp_low']}-{d['weather']['temp_high']}°C",
            "spots": [s["name"] for s in d["spots"]],
            "foods": [f["name"] for s in d["spots"] for f in s.get("nearby_foods", [])][:5],
        })

    prompt = f"""你是专业旅游规划师。以下是{city}从{start_date}到{end_date}的行程安排。

{json.dumps(days_summary, ensure_ascii=False, indent=2)}

请为每天行程生成解释（2-3句中文），内容包含：
1. 结合当天天气说明为何按此顺序安排景点（雨天优先室内，晴天优先户外）
2. 行程路线与附近美食的搭配建议

同时生成整体旅行总结（3-4句中文）。

以严格JSON格式输出，不要有其他文字：
{{
  "days": [{{"day": 1, "reasoning": "..."}}],
  "summary": "..."
}}"""

    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.content[0].text.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```")[0]
        return json.loads(raw.strip())
    except Exception:
        return {
            "days": [{"day": d["day"], "reasoning": "精彩行程，期待您的探索！"} for d in days_data],
            "summary": f"这是一次精彩的{city}之旅，祝您旅途愉快！",
        }


def plan_trip(request: TripRequest) -> TripResponse:
    dates = _parse_dates(request.start_date, request.end_date)
    total_days = len(dates)

    spot_map = get_spot_map(request.city)
    routes = get_popular_routes(request.city, total_days)
    all_foods = get_top_foods(request.city)

    raw_days = []
    for i, (date, route_ids) in enumerate(zip(dates, routes)):
        weather_dict = get_weather(request.city, date)

        # Look up spot details; skip unknown IDs
        day_spots = [spot_map[sid] for sid in route_ids if sid in spot_map]

        # Weather-based reordering
        day_spots = reorder_by_weather(day_spots, weather_dict["condition"])

        # Add transport info between consecutive spots
        day_spots = add_transport_info(day_spots)

        # Attach nearby foods to each spot
        for spot in day_spots:
            spot["nearby_foods"] = get_nearby_foods(spot, all_foods, limit=4)

        raw_days.append({
            "day": i + 1,
            "date": date,
            "weather": weather_dict,
            "spots": day_spots,
        })

    reasoning_data = _generate_reasoning(request.city, request.start_date, request.end_date, raw_days)
    reasoning_map = {d["day"]: d["reasoning"] for d in reasoning_data.get("days", [])}

    itinerary = []
    for d in raw_days:
        spots = []
        for s in d["spots"]:
            transport = None
            if t := s.get("transport_from_prev"):
                transport = TransportInfo(**t)
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
