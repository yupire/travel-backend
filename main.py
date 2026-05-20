from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import TripRequest, TripResponse
from agent import plan_trip
from tools.cities import get_cities

app = FastAPI(title="AI Travel Agent", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/cities")
def list_cities():
    """Mock: list of supported cities."""
    return {"cities": get_cities()}


@app.post("/plan", response_model=TripResponse)
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


@app.get("/health")
def health():
    return {"status": "ok"}
