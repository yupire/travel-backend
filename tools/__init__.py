import os

from langchain_core.tools import tool

from tools.cities import get_cities
from tools.spots import get_top_spots, geocode_spots
from tools.weather import get_weather
from tools.foods import get_top_foods
from tools.routes import get_popular_routes
from tools.clusters import cluster_spots_by_geo, export_clusters_to_geojson
from tools.directions import (
    plan_driving_route,
    plan_walking_route,
    plan_bicycling_route,
    plan_electrobike_route,
    plan_transit_route,
    plan_route,
)


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


@tool
def cluster_spots_geographically(
    city: str,
    limit: int = 20,
    eps_km: float = 3.0,
    min_samples: int = 2,
) -> list:
    """Group a city's tourist spots into geographic clusters using DBSCAN density-based clustering.

    Depends on the city spot query (get_tourist_spots). Use this AFTER you have a
    candidate set of attractions and want to organize them into day-trip regions,
    e.g. "scattered across 3 areas → 3 days, one area per day".

    Returns a list of clusters. Each cluster contains:
        cluster_id (int, -1 means noise/ungrouped), size, center {lat, lng},
        spread_km, indoor_count, outdoor_count, spot_ids, spot_names.
    """
    spots = get_top_spots(city, limit)
    return cluster_spots_by_geo(spots, eps_km=eps_km, min_samples=min_samples)


@tool
def export_clusters_geojson(
    city: str,
    limit: int = 20,
    eps_km: float = 3.0,
    min_samples: int = 2,
    output_path: str = "",
) -> dict:
    """Cluster city spots by geography and write the result to a GeoJSON file on disk.

    Same clustering logic as cluster_spots_geographically, but persists a
    GeoJSON FeatureCollection for mapping. Returns the FeatureCollection dict
    AND the resolved output_path. Pass output_path to choose where the file is
    written; default is backend/data/clusters_<city>.geojson.
    """
    spots = get_top_spots(city, limit)
    clusters = cluster_spots_by_geo(spots, eps_km=eps_km, min_samples=min_samples)
    if not output_path:
        output_path = os.path.join(
            os.path.dirname(__file__), "..", "data",
            f"clusters_{city.lower()}.geojson",
        )
    fc = export_clusters_to_geojson(clusters, output_path)
    return {"geojson": fc, "output_path": os.path.abspath(output_path)}


@tool
def plan_driving_directions(origin: str, destination: str) -> dict:
    """Plan driving route between two coordinates using Amap API.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)

    Returns:
        dict with origin, destination, taxi_cost, and paths list.
        Each path contains distance (meters) and step-by-step instructions.
    """
    return plan_driving_route(origin, destination)


@tool
def plan_walking_directions(origin: str, destination: str) -> dict:
    """Plan walking route between two coordinates using Amap API.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)

    Returns:
        dict with origin, destination, and paths list.
        Each path contains distance (meters), duration (seconds), and step-by-step instructions.
    """
    return plan_walking_route(origin, destination)


@tool
def plan_bicycling_directions(origin: str, destination: str) -> dict:
    """Plan bicycling route between two coordinates using Amap API.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)

    Returns:
        dict with origin, destination, and paths list.
        Each path contains distance (meters), duration (seconds), and step-by-step instructions.
    """
    return plan_bicycling_route(origin, destination)


@tool
def plan_electrobike_directions(origin: str, destination: str) -> dict:
    """Plan electric bicycle route between two coordinates using Amap API.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)

    Returns:
        dict with origin, destination, and paths list.
        Each path contains distance (meters), duration (seconds), and step-by-step instructions.
    """
    return plan_electrobike_route(origin, destination)


@tool
def plan_transit_directions(origin: str, destination: str, city: str) -> dict:
    """Plan public transit (bus/subway) route between two coordinates using Amap API.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)
        city: City name for transit planning (required for citycode parameter)

    Returns:
        dict with origin, destination, total distance, and transits list.
        Each transit contains walking_distance, nightflag, and segments (walking + bus/subway details).
    """
    return plan_transit_route(origin, destination, city)


@tool
def plan_route_directions(origin: str, destination: str, mode: str, city: str = "") -> dict:
    """Universal route planning tool supporting multiple transportation modes.

    Args:
        origin: Starting point coordinate as "lng,lat" (longitude first)
        destination: Ending point coordinate as "lng,lat" (longitude first)
        mode: Transportation mode: "driving", "walking", "bicycling", "electrobike", or "transit"
        city: City name (required for transit mode, optional for others)

    Returns:
        dict with route details depending on the mode (distance, duration, steps, segments, etc.)
    """
    if mode == "transit" and not city:
        return {"error": "city parameter is required for transit mode"}
    if mode == "transit":
        return plan_route(origin, destination, mode, city)
    return plan_route(origin, destination, mode)


# Registry of all LangChain tools — bind to an LLM with llm.bind_tools(ALL_TOOLS)
ALL_TOOLS = [
    list_supported_cities,
    get_tourist_spots,
    geocode_spot_locations,
    get_city_weather,
    get_food_recommendations,
    get_trip_routes,
    cluster_spots_geographically,
    export_clusters_geojson,
    plan_driving_directions,
    plan_walking_directions,
    plan_bicycling_directions,
    plan_electrobike_directions,
    plan_transit_directions,
    plan_route_directions,
]
