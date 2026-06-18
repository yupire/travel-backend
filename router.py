from datetime import datetime

from fastapi import APIRouter, HTTPException

from models import TripRequest, TripResponse
from agent import plan_trip
from tools.cities import get_cities,get_city
from tools import weather
from tools.spots import get_spot_map

router = APIRouter()


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
    try:
        start = datetime.strptime(request.start_date, "%Y-%m-%d")
        end = datetime.strptime(request.end_date, "%Y-%m-%d")
    except ValueError:
        raise HTTPException(status_code=400, detail="日期格式错误，请使用 YYYY-MM-DD")

    if end < start:
        raise HTTPException(status_code=400, detail="结束日期不能早于开始日期")

    days = (end - start).days + 1
    if days > 7:
        raise HTTPException(status_code=400, detail="行程最长支持7天")

    try:
        return plan_trip(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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