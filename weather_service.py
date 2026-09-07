"""Weather data layer for WeatherGPT.

Wraps the OpenWeatherMap APIs and returns plain, JSON-serialisable dicts.

Design rules for this module:
  * Never raise on an expected failure. Return ``{"error": "..."}`` instead,
    so the caller (and ultimately the LLM) always receives structured data.
  * Never invent data. If a field is missing upstream it stays missing here.
  * Never include the API key in an error message or log line.

Endpoint choice matters for cost. Geocoding, current weather and the
5-day/3-hour forecast are all on OpenWeatherMap's free tier. Government
weather alerts are only exposed by the One Call 3.0 endpoint, which needs a
separate (free-to-start but card-backed) subscription. ``get_alerts`` therefore
degrades honestly: if One Call is not available on the key it reports that
alerts could not be checked rather than claiming there are none.
"""

from __future__ import annotations

import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

import config

load_dotenv()

GEOCODE_URL = "https://api.openweathermap.org/geo/1.0/direct"
CURRENT_URL = "https://api.openweathermap.org/data/2.5/weather"
FORECAST_URL = "https://api.openweathermap.org/data/2.5/forecast"
ONECALL_URL = "https://api.openweathermap.org/data/3.0/onecall"

REQUEST_TIMEOUT = 10  # seconds
MAX_FORECAST_DAYS = 5  # limit of the free 5-day/3-hour endpoint

# Geocoding results are stable, so cache them for the life of the process.
_geocode_cache: Dict[str, Dict[str, Any]] = {}


def _api_key() -> Optional[str]:
    key = os.getenv("OPENWEATHER_API_KEY", "").strip()
    return key if config.is_configured(key) else None


def _missing_key_error() -> Dict[str, str]:
    return {
        "error": "OpenWeatherMap API key is not configured. "
                 "Add OPENWEATHER_API_KEY to your .env file."
    }


def _request(url: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Perform one GET request and normalise every failure mode.

    Returns ``{"ok": True, "data": ...}`` or ``{"ok": False, "error": ...}``.
    The ``status`` key is included on HTTP errors so callers can react to
    specific codes (e.g. One Call's 401 meaning "not subscribed").
    """
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    except requests.exceptions.Timeout:
        return {"ok": False,
                "error": "The weather service took too long to respond. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"ok": False,
                "error": "Could not reach the weather service. Check your internet connection."}
    except requests.exceptions.RequestException:
        return {"ok": False, "error": "The weather request failed unexpectedly."}

    if response.status_code == 401:
        return {"ok": False, "status": 401,
                "error": "The OpenWeatherMap API key was rejected. "
                         "Check that OPENWEATHER_API_KEY is valid and active."}
    if response.status_code == 404:
        return {"ok": False, "status": 404,
                "error": "The weather service has no data for that location."}
    if response.status_code == 429:
        return {"ok": False, "status": 429,
                "error": "The weather service rate limit was exceeded. "
                         "Please wait a moment and try again."}
    if response.status_code >= 500:
        return {"ok": False, "status": response.status_code,
                "error": "The weather service is temporarily unavailable."}
    if response.status_code != 200:
        return {"ok": False, "status": response.status_code,
                "error": "The weather service returned an unexpected status "
                         f"({response.status_code})."}

    try:
        return {"ok": True, "data": response.json()}
    except ValueError:
        return {"ok": False, "error": "The weather service returned a malformed response."}


def _describe_place(place: Dict[str, Any]) -> str:
    """Build a human label like 'Delhi, DL, IN' from a geocoding record."""
    parts = [place.get("name"), place.get("state"), place.get("country")]
    return ", ".join(str(p) for p in parts if p)


def geocode_location(city_name: str) -> Dict[str, Any]:
    """Resolve a free-text place name to coordinates.

    Returns ``{"name", "country", "state", "lat", "lon", "label"}``
    or ``{"error": ...}``.
    """
    if not isinstance(city_name, str) or not city_name.strip():
        return {"error": "No location was provided. Please name a city."}

    query = city_name.strip()
    cache_key = query.lower()
    if cache_key in _geocode_cache:
        return dict(_geocode_cache[cache_key])

    key = _api_key()
    if key is None:
        return _missing_key_error()

    result = _request(GEOCODE_URL, {"q": query, "limit": 1, "appid": key})
    if not result["ok"]:
        return {"error": result["error"]}

    matches = result["data"]
    if not isinstance(matches, list) or not matches:
        return {"error": f"Could not find location: {query}"}

    place = matches[0]
    if not isinstance(place, dict) or "lat" not in place or "lon" not in place:
        return {"error": f"Could not find location: {query}"}

    resolved = {
        "name": place.get("name", query),
        "country": place.get("country"),
        "state": place.get("state"),
        "lat": place["lat"],
        "lon": place["lon"],
        "label": _describe_place(place),
    }
    _geocode_cache[cache_key] = resolved
    return dict(resolved)


def get_current_weather(location: str, units: str = "metric") -> Dict[str, Any]:
    """Current conditions for a named location."""
    place = geocode_location(location)
    if "error" in place:
        return place

    key = _api_key()
    if key is None:
        return _missing_key_error()

    result = _request(CURRENT_URL, {
        "lat": place["lat"], "lon": place["lon"], "appid": key, "units": units,
    })
    if not result["ok"]:
        return {"error": result["error"]}

    data = result["data"]
    if not isinstance(data, dict) or "main" not in data:
        return {"error": f"The weather service returned no usable data for {place['label']}."}

    main = data.get("main") or {}
    wind = data.get("wind") or {}
    weather_list = data.get("weather") or []
    condition = weather_list[0] if weather_list and isinstance(weather_list[0], dict) else {}

    return {
        "location": place["label"],
        "units": _unit_labels(units),
        "condition": condition.get("main"),
        "description": condition.get("description"),
        "temperature": main.get("temp"),
        "feels_like": main.get("feels_like"),
        "temp_min": main.get("temp_min"),
        "temp_max": main.get("temp_max"),
        "humidity": main.get("humidity"),
        "pressure": main.get("pressure"),
        "wind_speed": wind.get("speed"),
        "wind_deg": wind.get("deg"),
        "cloudiness": (data.get("clouds") or {}).get("all"),
        "visibility": data.get("visibility"),
        "observed_at": _local_time(data.get("dt"), data.get("timezone")),
    }


def get_forecast(location: str, days: int = 3, units: str = "metric") -> Dict[str, Any]:
    """Daily forecast, aggregated from the free 3-hourly endpoint."""
    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 3
    days = max(1, min(days, MAX_FORECAST_DAYS))

    place = geocode_location(location)
    if "error" in place:
        return place

    key = _api_key()
    if key is None:
        return _missing_key_error()

    result = _request(FORECAST_URL, {
        "lat": place["lat"], "lon": place["lon"], "appid": key, "units": units,
    })
    if not result["ok"]:
        return {"error": result["error"]}

    data = result["data"]
    entries = data.get("list") if isinstance(data, dict) else None
    if not isinstance(entries, list) or not entries:
        return {"error": f"No forecast data is available for {place['label']}."}

    offset = (data.get("city") or {}).get("timezone", 0) or 0
    daily = _aggregate_daily(entries, offset)
    if not daily:
        return {"error": f"No forecast data is available for {place['label']}."}

    return {
        "location": place["label"],
        "units": _unit_labels(units),
        "days_requested": days,
        "forecast": daily[:days],
    }


def _day_label(day_iso: str, today_iso: str) -> str:
    """Name a forecast day relative to today, in the location's own time.

    Without this the model has only a bare date and has been observed calling
    today's entry "tomorrow". A label it cannot misread removes the guesswork.
    """
    try:
        day = datetime.fromisoformat(day_iso).date()
        today = datetime.fromisoformat(today_iso).date()
    except ValueError:
        return day_iso
    delta = (day - today).days
    if delta == 0:
        return "today"
    if delta == 1:
        return "tomorrow"
    if 2 <= delta <= 6:
        return day.strftime("%A")
    return day_iso


def _aggregate_daily(entries: List[Dict[str, Any]], tz_offset: int) -> List[Dict[str, Any]]:
    """Collapse 3-hourly slots into one summary per local calendar day."""
    buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    order: List[str] = []
    # "Today" means today where the weather is, not where the server is.
    today_iso = (datetime.now(timezone.utc) + timedelta(seconds=tz_offset)).date().isoformat()

    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("dt"), (int, float)):
            continue
        local = datetime.fromtimestamp(entry["dt"], tz=timezone.utc) + timedelta(seconds=tz_offset)
        day = local.date().isoformat()
        if day not in buckets:
            order.append(day)
        buckets[day].append({"entry": entry, "hour": local.hour})

    summaries = []
    for day in order:
        slots = buckets[day]
        temps = [
            (s["entry"].get("main") or {}).get("temp")
            for s in slots
            if isinstance((s["entry"].get("main") or {}).get("temp"), (int, float))
        ]
        if not temps:
            continue

        pops = [
            s["entry"].get("pop") for s in slots
            if isinstance(s["entry"].get("pop"), (int, float))
        ]
        # Describe the day from a midday slot when one exists: a 3am reading is
        # a poor summary of what the day will actually feel like.
        midday = min(slots, key=lambda s: abs(s["hour"] - 12))
        weather_list = midday["entry"].get("weather") or []
        condition = weather_list[0] if weather_list and isinstance(weather_list[0], dict) else {}

        summaries.append({
            "date": day,
            "day_label": _day_label(day, today_iso),
            "temp_min": round(min(temps), 1),
            "temp_max": round(max(temps), 1),
            "condition": condition.get("main"),
            "description": condition.get("description"),
            "chance_of_rain_percent": round(max(pops) * 100) if pops else None,
            "humidity": (midday["entry"].get("main") or {}).get("humidity"),
            "wind_speed": (midday["entry"].get("wind") or {}).get("speed"),
        })

    return summaries


def get_alerts(location: str) -> Dict[str, Any]:
    """Active government weather alerts, via One Call 3.0.

    One Call is a paid add-on. When it is unavailable this reports
    ``alerts_available: False`` rather than an empty alert list, so the
    assistant never tells a user "no alerts" when it simply could not look.
    """
    place = geocode_location(location)
    if "error" in place:
        return place

    key = _api_key()
    if key is None:
        return _missing_key_error()

    result = _request(ONECALL_URL, {
        "lat": place["lat"], "lon": place["lon"], "appid": key,
        "exclude": "minutely,hourly,daily,current",
    })

    if not result["ok"]:
        if result.get("status") in (401, 403):
            return {
                "location": place["label"],
                "alerts_available": False,
                "alerts": [],
                "note": "Weather alerts require an OpenWeatherMap One Call 3.0 "
                        "subscription, which this API key does not have. Alerts "
                        "could not be checked for this location.",
            }
        return {"error": result["error"]}

    data = result["data"]
    raw_alerts = data.get("alerts", []) if isinstance(data, dict) else []
    alerts = []
    for alert in raw_alerts if isinstance(raw_alerts, list) else []:
        if not isinstance(alert, dict):
            continue
        alerts.append({
            "event": alert.get("event"),
            "sender": alert.get("sender_name"),
            "starts": _local_time(alert.get("start"), data.get("timezone_offset")),
            "ends": _local_time(alert.get("end"), data.get("timezone_offset")),
            "description": (alert.get("description") or "")[:600],
            "tags": alert.get("tags") or [],
        })

    return {
        "location": place["label"],
        "alerts_available": True,
        "alert_count": len(alerts),
        "alerts": alerts,
    }


def _unit_labels(units: str) -> Dict[str, str]:
    if units == "imperial":
        return {"temperature": "°F", "wind_speed": "mph"}
    if units == "standard":
        return {"temperature": "K", "wind_speed": "m/s"}
    return {"temperature": "°C", "wind_speed": "m/s"}


def _local_time(dt_value: Any, tz_offset: Any) -> Optional[str]:
    """Format a UNIX timestamp in the location's own local time."""
    if not isinstance(dt_value, (int, float)):
        return None
    offset = tz_offset if isinstance(tz_offset, (int, float)) else 0
    local = datetime.fromtimestamp(dt_value, tz=timezone.utc) + timedelta(seconds=offset)
    return local.strftime("%Y-%m-%d %H:%M local")
