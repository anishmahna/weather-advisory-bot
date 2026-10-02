"""Deterministic Open-Meteo client. NO LLM in this file. Only source of numbers."""
from datetime import date, timedelta
import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10


class WeatherError(Exception):
    kind = "weather_unavailable"


class LocationUnresolved(WeatherError):
    kind = "location_unresolved"


class WeatherUnavailable(WeatherError):
    kind = "weather_unavailable"


def geocode(city: str) -> dict:
    try:
        r = requests.get(GEO_URL, params={"name": city, "count": 5, "language": "en", "format": "json"}, timeout=TIMEOUT)
        r.raise_for_status()
        results = r.json().get("results")
    except (requests.RequestException, ValueError) as e:
        raise WeatherUnavailable(f"geocoding failed: {e}")
    if not results:
        raise LocationUnresolved(city)
    first = results[0]    # documented default: first candidate
    display = ", ".join(x for x in [first.get("name"), first.get("admin1"), first.get("country")] if x)
    return {"lat": first["latitude"], "lon": first["longitude"], "display": display, "candidates": len(results)}


def fetch_forecast(lat: float, lon: float, fields: dict) -> dict:
    params = {
        "latitude": lat, "longitude": lon, "timezone": "auto",
        "current": ",".join(fields["current"]),
        "hourly": ",".join(fields["hourly"]),
        "daily": ",".join(fields["daily"]),
        "forecast_days": fields.get("forecast_days", 3),
    }
    try:
        r = requests.get(FORECAST_URL, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        raw = r.json()
    except (requests.RequestException, ValueError) as e:
        raise WeatherUnavailable(f"forecast failed: {e}")
    if not isinstance(raw, dict) or "current" not in raw or "hourly" not in raw or "daily" not in raw:
        raise WeatherUnavailable("forecast response missing expected sections")
    return raw


def build_snapshot(raw: dict, time_window: str, taxonomy: dict, fields: dict) -> dict:
    """Pick the values for the requested time. Everything comes from `raw`."""
    cur = raw["current"]
    now_ts = cur["time"]
    use_current = time_window in ("now", "today")
    if use_current:
        ts = now_ts[:13] + ":00"
    else:
        day = date.fromisoformat(now_ts[:10])
        if time_window == "tomorrow":
            day += timedelta(days=1)
        hour = taxonomy["representative_hours"][time_window]
        ts = f"{day.isoformat()}T{hour:02d}:00"
    htimes = raw["hourly"]["time"]
    if ts not in htimes:
        raise WeatherUnavailable(f"forecast does not cover {ts}")
    hi = htimes.index(ts)
    values, units = {}, {}
    for f in fields["hourly"]:
        series = raw["hourly"].get(f)
        values[f] = series[hi] if series else None
        units[f] = raw.get("hourly_units", {}).get(f, "")
    if use_current:                       # live "current" values override hourly where we asked for them
        for f in fields["current"]:
            if cur.get(f) is not None:
                values[f] = cur[f]
                units[f] = raw.get("current_units", {}).get(f, units.get(f, ""))
    dtimes = raw["daily"]["time"]
    di = dtimes.index(ts[:10]) if ts[:10] in dtimes else None
    for f in fields["daily"]:
        series = raw["daily"].get(f)
        values[f"daily_{f}"] = series[di] if (series and di is not None) else None
        units[f"daily_{f}"] = raw.get("daily_units", {}).get(f, "")
    values["local_hour"] = int(ts[11:13])
    units["local_hour"] = "h"
    return {"values": values, "units": units, "time": ts, "window": time_window}
