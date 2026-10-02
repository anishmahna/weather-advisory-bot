"""Offline eval helpers: Open-Meteo-shaped synthetic payloads, fake HTTP, scripted LLM.
NOTE: payloads here are SYNTHETIC (shaped like real API responses), not live captures."""
import json
from datetime import date, timedelta
import requests

HOURLY_DEFAULT = {
    "temperature_2m": 27.0, "apparent_temperature": 29.0, "relative_humidity_2m": 60,
    "precipitation_probability": 10, "precipitation": 0.0, "weather_code": 1,
    "pressure_msl": 1008.0, "wind_speed_10m": 10.0, "wind_gusts_10m": 18.0,
    "uv_index": lambda ts: round(max(0, 5 - abs(int(ts[11:13]) - 13) * 0.8), 1),
}
DAILY_DEFAULT = {"precipitation_sum": 2.0, "precipitation_probability_max": 20,
                 "wind_gusts_10m_max": 20.0, "uv_index_max": 5.0, "temperature_2m_max": 30.0}
UNITS = {"temperature_2m": "°C", "apparent_temperature": "°C", "relative_humidity_2m": "%",
         "precipitation_probability": "%", "precipitation": "mm", "weather_code": "wmo code",
         "pressure_msl": "hPa", "wind_speed_10m": "km/h", "wind_gusts_10m": "km/h", "uv_index": ""}
DUNITS = {"precipitation_sum": "mm", "precipitation_probability_max": "%", "wind_gusts_10m_max": "km/h",
          "uv_index_max": "", "temperature_2m_max": "°C"}


def make_payload(now="2026-09-04T10:00", hourly=None, daily=None, current=None):
    hourly, daily, current = hourly or {}, daily or {}, current or {}
    d0 = date.fromisoformat(now[:10])
    times = [f"{(d0 + timedelta(days=d)).isoformat()}T{h:02d}:00" for d in range(3) for h in range(24)]

    def val(spec, ts):
        return spec(ts) if callable(spec) else spec
    h = {k: [val(hourly.get(k, v), t) for t in times] for k, v in HOURLY_DEFAULT.items()}
    i = times.index(now[:13] + ":00")
    cur = {"time": now, **{k: h[k][i] for k in HOURLY_DEFAULT if k != "precipitation_probability"}, **current}
    d = {k: [daily.get(k, v)] * 3 for k, v in DAILY_DEFAULT.items()}
    return {"current": cur, "current_units": UNITS, "hourly": {"time": times, **h}, "hourly_units": UNITS,
            "daily": {"time": [(d0 + timedelta(days=n)).isoformat() for n in range(3)], **d}, "daily_units": DUNITS}


class FakeResp:
    def __init__(self, data): self._d = data
    def raise_for_status(self): pass
    def json(self): return self._d


def install_http(monkeypatch, payload=None, geo=None, down=False, calls=None):
    geo = [{"name": "Bhopal", "admin1": "Madhya Pradesh", "country": "India",
            "latitude": 23.25, "longitude": 77.4}] if geo is None else geo

    def fake_get(url, params=None, timeout=None):
        if calls is not None:
            calls.append((url, dict(params or {})))
        if down:
            raise requests.ConnectionError("simulated outage")
        return FakeResp({"results": geo} if "geocoding" in url else payload)
    monkeypatch.setattr(requests, "get", fake_get)


class Script:
    """Scripted LLM. Routes on the [TASK:...] tag at the top of each system prompt."""
    def __init__(self, intake=None, fuzzy=None, compose="faithful"):
        self.intake, self.fuzzy, self.compose = intake, fuzzy, compose

    def __call__(self, system, user):
        if "[TASK:intake]" in system:
            i = self.intake(user) if callable(self.intake) else self.intake
            return json.dumps(i)
        if "[TASK:fuzzy]" in system:
            return json.dumps({"matches": self.fuzzy or []})
        if "[TASK:compose]" in system:
            return faithful_compose(json.loads(user)) if self.compose == "faithful" else self.compose
        raise AssertionError("unknown task")


def faithful_compose(p):
    """Stand-in for a well-behaved model: only uses provided facts and SOP text."""
    parts = []
    for s in p["lead_sops"] or [p["primary_sop"]]:
        parts.append(f"[{s['id']}] {s['title']}: {s['advice']}")
    parts.append("Readings: " + "; ".join(p["facts"][2:]))
    for s in p["additional_sops"]:
        parts.append(f"Also [{s['id']}]: {s['advice']}")
    return "\n".join(parts)


def intent(activity="cycling", location="Bhopal", audiences=None, window="today", summary="asks about activity", outdoor=True):
    return {"outdoor_related": outdoor, "location": location, "activity": activity,
            "audiences": audiences or [], "time_window": window, "intent_summary": summary}
