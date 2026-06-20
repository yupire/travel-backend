"""输出解析与逐字段兜底

把 LLM / 降级子图产出的最终消息收敛成一个合法、完整的 TripResponse：
- _extract_json_payload : 从脏文本里尽力提取 JSON 对象
- _to_* / _coerce_*     : 逐字段宽容转换，缺失或类型错就填默认值而非整天丢弃
- _build_trip_response  : 顶层补齐 + 逐天兜底，绝不抛异常（保证 /plan 不 500）
"""
from __future__ import annotations

import json
import logging

from models import (
    DayPlan,
    FoodRec,
    SpotPlan,
    TransportInfo,
    TripRequest,
    TripResponse,
    WeatherInfo,
)

logger = logging.getLogger(__name__)


def _extract_json_payload(content) -> dict | None:
    """从 LLM 最终消息中尽力提取 JSON 对象。

    容忍三种常见脏输出：
    1. content 是 list[dict]（部分模型分块返回）；
    2. 用 ```json ... ``` 代码块包裹；
    3. JSON 前后夹带解释性文字。
    解析失败返回 None。
    """
    if isinstance(content, list):
        content = "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    text = str(content or "").strip()
    if not text:
        return None

    # 去掉 ```json ... ``` 代码块包裹
    if text.startswith("```"):
        text = text.strip("`").strip()
        nl = text.find("\n")
        if nl != -1 and text[:nl].strip().lower() in ("json", ""):
            text = text[nl + 1:].strip()

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except (TypeError, ValueError):
        pass

    # 截取第一个 { 到最后一个 }，容忍前后多余文字
    l, r = text.find("{"), text.rfind("}")
    if l != -1 and r > l:
        try:
            obj = json.loads(text[l:r + 1])
            return obj if isinstance(obj, dict) else None
        except (TypeError, ValueError):
            return None
    return None


# ──────────────────────── 逐字段兜底（coercion）────────────────────────
# 设计原则：LLM 输出常常缺字段 / 类型错（如 temp_high 给了字符串、spots 少了 lat）。
# 直接 DayPlan(**day) 会因单个字段不合法而整天丢弃，导致「行程为空」。这里改为
# 逐字段宽容转换：缺失或不合法就填合理默认值，最大限度保住每一天 / 每一个景点。
def _to_int(v, default: int = 0) -> int:
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def _to_float(v, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _to_str(v, default: str = "") -> str:
    if v is None:
        return default
    return v if isinstance(v, str) else str(v)


def _to_bool(v, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in ("true", "1", "yes", "indoor", "室内")
    if isinstance(v, (int, float)):
        return bool(v)
    return default


def _to_str_list(v) -> list[str]:
    if isinstance(v, list):
        return [_to_str(x) for x in v if x is not None]
    if isinstance(v, str) and v.strip():
        return [v.strip()]
    return []


def _coerce_food(raw, idx: int) -> dict | None:
    """单条美食兜底；完全无法解析（非 dict）则丢弃返回 None。"""
    if not isinstance(raw, dict):
        return None
    return FoodRec(
        id=_to_str(raw.get("id"), f"food_{idx}"),
        name=_to_str(raw.get("name"), "未知美食"),
        cuisine=_to_str(raw.get("cuisine"), "其他"),
        price_range=_to_str(raw.get("price_range"), "$$"),
        rating=_to_float(raw.get("rating"), 0.0),
        distance_m=_to_int(raw.get("distance_m"), 0),
    ).model_dump()


def _coerce_transport(raw) -> dict | None:
    """交通信息兜底；缺失（None / 非 dict）则返回 None（schema 允许 null）。"""
    if not isinstance(raw, dict):
        return None
    return TransportInfo(
        mode=_to_str(raw.get("mode"), "walking"),
        duration_min=_to_int(raw.get("duration_min"), 0),
        cost=_to_float(raw.get("cost"), 0.0),
        distance_km=_to_float(raw.get("distance_km"), 0.0),
    ).model_dump()


def _coerce_spot(raw, idx: int) -> dict | None:
    """单个景点逐字段兜底；非 dict 无法挽救则丢弃返回 None。"""
    if not isinstance(raw, dict):
        return None
    spot_type = _to_str(raw.get("type"), "outdoor")
    # is_indoor 缺失时从 type 推断，避免室内外信息丢失
    is_indoor = (
        _to_bool(raw.get("is_indoor"))
        if raw.get("is_indoor") is not None
        else ("indoor" in spot_type.lower() or "室内" in spot_type)
    )
    foods = [
        f for f in (
            _coerce_food(x, i) for i, x in enumerate(raw.get("nearby_foods") or [])
        ) if f is not None
    ]
    return SpotPlan(
        id=_to_str(raw.get("id"), f"spot_{idx}"),
        name=_to_str(raw.get("name"), "未知景点"),
        lat=_to_float(raw.get("lat"), 0.0),
        lng=_to_float(raw.get("lng"), 0.0),
        duration_min=_to_int(raw.get("duration_min"), 60),
        open_time=_to_str(raw.get("open_time"), ""),
        ticket=_to_float(raw.get("ticket"), 0.0),
        type=spot_type,
        is_indoor=is_indoor,
        tags=_to_str_list(raw.get("tags")),
        description=_to_str(raw.get("description"), ""),
        nearby_foods=foods,
        transport_from_prev=_coerce_transport(raw.get("transport_from_prev")),
    ).model_dump()


def _coerce_weather(raw) -> dict:
    """天气逐字段兜底；缺失整体也返回一个合法 WeatherInfo。"""
    raw = raw if isinstance(raw, dict) else {}
    return WeatherInfo(
        condition=_to_str(raw.get("condition"), "unknown"),
        temp_high=_to_int(raw.get("temp_high"), 0),
        temp_low=_to_int(raw.get("temp_low"), 0),
        description=_to_str(raw.get("description"), ""),
    ).model_dump()


def _coerce_day(raw, idx: int, dates: list[str]) -> dict:
    """单天逐字段兜底，永远返回一个合法 DayPlan dict（不再整天丢弃）。"""
    raw = raw if isinstance(raw, dict) else {}
    default_date = dates[idx - 1] if 0 < idx <= len(dates) else ""
    spots = [
        s for s in (
            _coerce_spot(x, i) for i, x in enumerate(raw.get("spots") or [], start=1)
        ) if s is not None
    ]
    return DayPlan(
        day=_to_int(raw.get("day"), idx),
        date=_to_str(raw.get("date"), default_date),
        weather=_coerce_weather(raw.get("weather")),
        spots=spots,
        reasoning=_to_str(raw.get("reasoning"), ""),
        is_indoor_outdoor_filter=_to_bool(raw.get("is_indoor_outdoor_filter")),
    ).model_dump()


def _build_trip_response(
    payload: dict | None,
    request: TripRequest,
    dates: list[str],
    raw_text: str,
) -> TripResponse:
    """把任意解析结果强制收敛成一个合法、完整的 TripResponse，绝不抛异常。

    - 顶层必填标量缺失 → 用请求参数补齐；
    - itinerary 逐天逐字段兜底补全：缺字段填默认值而非整天丢弃，最大限度保住行程；
    - summary 缺失 → 用 reasoning 文本 / 原始输出兜底。
    这样即便 Agent 没能给出规范行程，/plan 也总能返回结构完整的 JSON（而非 500）。
    """
    from pydantic import ValidationError

    payload = dict(payload) if isinstance(payload, dict) else {}

    payload.setdefault("city", request.city)
    payload.setdefault("start_date", request.start_date)
    payload.setdefault("end_date", request.end_date)
    payload.setdefault("total_days", len(dates))

    # 逐天逐字段兜底：每一天都尽量保住，只有彻底无法构造（极少见）才记录并跳过
    raw_itinerary = payload.get("itinerary")
    valid_days: list[dict] = []
    if isinstance(raw_itinerary, list):
        for i, day in enumerate(raw_itinerary, start=1):
            try:
                valid_days.append(_coerce_day(day, i, dates))
            except (ValidationError, TypeError) as e:
                logger.warning("itinerary 第 %d 天兜底构造仍失败，已跳过：%s", i, e)
    else:
        logger.warning("itinerary 不是数组（实际类型=%s），无行程可解析",
                       type(raw_itinerary).__name__)
    payload["itinerary"] = valid_days

    # summary 兜底：兼容 reasoning 残缺 schema 里的 days 文本
    if not payload.get("summary"):
        days_text = str(payload["days"]) if payload.get("days") else ""
        payload["summary"] = days_text or raw_text or "（未能生成行程说明）"

    try:
        resp = TripResponse(**payload)
        if not valid_days:
            logger.warning("plan_trip 输出无有效行程天，仅返回 summary 兜底结构")
        return resp
    except ValidationError as e:
        logger.error("TripResponse 最终校验仍失败，返回最小安全结构：%s", e)
        return TripResponse(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            total_days=len(dates),
            itinerary=[],
            summary=payload.get("summary") or raw_text or "（无输出）",
        )
