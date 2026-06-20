import json
from datetime import datetime

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import TripRequest, TripResponse
from agent import plan_trip, plan_trip_stream
from tools.cities import get_cities,get_city
from tools import weather
from tools.spots import get_spot_map

router = APIRouter()


def _validate_plan_request(request: TripRequest) -> None:
    """校验行程请求的日期合法性，不合法时抛 HTTPException。"""
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD")

    if end < start:
        raise HTTPException(status_code=400, detail="结束日期不能早于开始日期")

    if (end - start).days + 1 > 7:
        raise HTTPException(status_code=400, detail="行程最长支持7天")


# 城市列表
@router.get("/cities")
def list_cities():
    """返回支持的城市列表。"""
    return {"cities": get_cities()}

# 查询单个城市
@router.get("/cities/{location}")
def list_cities(location:str):
    """按名称查询单个城市。"""
    return {"cities": get_city(location)}


# 生成行程
@router.post("/plan", response_model=TripResponse)
def create_plan(request: TripRequest):
    _validate_plan_request(request)
    try:
        return plan_trip(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 生成行程（流式）
# 以 NDJSON 流式返回：Agent 推理 + 工具调用完成后先吐一条 progress 事件
# （前端展示「规划已完成，正在整理」），format 结构化完成后再吐 result 事件。
@router.post("/plan/stream")
def create_plan_stream(request: TripRequest):
    # 日期校验在开始流式响应前完成：不合法直接返回 4xx，避免在流中途报错。
    _validate_plan_request(request)

    def event_stream():
        try:
            for event in plan_trip_stream(request):
                yield json.dumps(event, ensure_ascii=False) + "\n"
        except Exception as e:
            yield json.dumps(
                {"type": "error", "detail": str(e)}, ensure_ascii=False
            ) + "\n"

    return StreamingResponse(
        event_stream(),
        media_type="application/x-ndjson",
        # 关闭 Nginx 等反向代理的缓冲，确保进度事件能即时送达前端
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


# 健康检查
@router.get("/health")
def health():
    return {"status": "ok"}


# 外部api调用
@router.get("/weather")
def get_weather(city: str, date: str):
    print(f"API 请求天气 city={city} date={date}")
    return weather.get_weather(city, date)   

@router.get("/spots")
def get_spots(city: str, limit: int = 10):
    return get_spot_map(city)