from langchain_core.tools import tool

from tools.cities import get_cities
from tools.spots import get_top_spots, geocode_spots
from tools.weather import get_weather
from tools.foods import get_top_foods
from tools.routes import get_popular_routes


@tool
def list_supported_cities() -> list:
    """Returns all supported travel destination cities with id, name, and country."""
    return get_cities()


@tool
def get_tourist_spots(city: str, limit: int = 10) -> list:
    """Get top tourist spots for a city. city must be lowercase (e.g. 'tokyo', 'paris', 'singapore')."""
    return get_top_spots(city, limit)


@tool
def geocode_spot_locations(city: str, spot_names: list) -> list:
    """Resolve spot names to coordinates (lat, lng) and indoor/outdoor attribute for a city.

    Use after picking candidate attractions to attach map locations before
    daily route construction. Returns [{id, name, lat, lng, is_indoor}];
    unknown names are silently skipped.
    """
    return geocode_spots(city, spot_names)


@tool
def get_city_weather(city: str, date: str) -> dict:
    """Get weather forecast for a city on a given date. date format: YYYY-MM-DD."""
    return get_weather(city, date)


@tool
def get_food_recommendations(city: str, limit: int = 10) -> list:
    """Get top food and restaurant recommendations for a city."""
    return get_top_foods(city, limit)


@tool
def get_trip_routes(city: str, days: int) -> list:
    """Get popular pre-planned spot sequences for a city trip of N days. Returns spot IDs per day."""
    return get_popular_routes(city, days)


# Registry of all LangChain tools — bind to an LLM with llm.bind_tools(ALL_TOOLS)
ALL_TOOLS = [
    list_supported_cities,
    get_tourist_spots,
    geocode_spot_locations,
    get_city_weather,
    get_food_recommendations,
    get_trip_routes,
]
