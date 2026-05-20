from pydantic import BaseModel
from typing import List, Optional


class TripRequest(BaseModel):
    city: str
    start_date: str  # "2024-06-01"
    end_date: str    # "2024-06-03"


class WeatherInfo(BaseModel):
    condition: str   # sunny | partly_cloudy | cloudy | rainy
    temp_high: int
    temp_low: int
    description: str


class FoodRec(BaseModel):
    id: str
    name: str
    cuisine: str
    price_range: str  # $ | $$ | $$$ | $$$$
    rating: float
    distance_m: int


class TransportInfo(BaseModel):
    mode: str         # walking | subway | bus | taxi
    duration_min: int
    cost: float
    distance_km: float


class SpotPlan(BaseModel):
    id: str
    name: str
    lat: float
    lng: float
    duration_min: int
    open_time: str
    ticket: float
    type: str         # indoor | outdoor
    tags: List[str]
    description: str
    nearby_foods: List[FoodRec] = []
    transport_from_prev: Optional[TransportInfo] = None


class DayPlan(BaseModel):
    day: int
    date: str
    weather: WeatherInfo
    spots: List[SpotPlan]
    reasoning: str


class TripResponse(BaseModel):
    city: str
    start_date: str
    end_date: str
    total_days: int
    itinerary: List[DayPlan]
    summary: str
