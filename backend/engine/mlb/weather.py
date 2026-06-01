"""
Open-Meteo weather integration for MLB stadium conditions.
No API key required — Open-Meteo is a free, open-source weather API.

Live forecast: https://api.open-meteo.com/v1/forecast
Historical:    https://archive-api.open-meteo.com/v1/archive
"""

import math
from datetime import date, timedelta
from typing import Optional

import requests

# MLB stadium coordinates: (latitude, longitude, IANA timezone)
STADIUM_COORDS: dict[str, tuple[float, float, str]] = {
    "AZ":  (33.4453, -112.0667, "America/Phoenix"),
    "ATL": (33.8907,  -84.4678, "America/New_York"),
    "BAL": (39.2838,  -76.6218, "America/New_York"),
    "BOS": (42.3467,  -71.0972, "America/New_York"),
    "CHC": (41.9484,  -87.6553, "America/Chicago"),
    "CWS": (41.8300,  -87.6338, "America/Chicago"),
    "CIN": (39.0979,  -84.5082, "America/New_York"),
    "CLE": (41.4959,  -81.6854, "America/New_York"),
    "COL": (39.7559, -104.9942, "America/Denver"),
    "DET": (42.3390,  -83.0485, "America/Detroit"),
    "HOU": (29.7572,  -95.3555, "America/Chicago"),
    "KC":  (39.0517,  -94.4803, "America/Chicago"),
    "LAA": (33.8003, -117.8827, "America/Los_Angeles"),
    "LAD": (34.0739, -118.2400, "America/Los_Angeles"),
    "MIA": (25.7781,  -80.2197, "America/New_York"),
    "MIL": (43.0280,  -87.9712, "America/Chicago"),
    "MIN": (44.9817,  -93.2778, "America/Chicago"),
    "NYM": (40.7571,  -73.8458, "America/New_York"),
    "NYY": (40.8296,  -73.9262, "America/New_York"),
    "ATH": (37.7516, -122.2005, "America/Los_Angeles"),
    "PHI": (39.9061,  -75.1665, "America/New_York"),
    "PIT": (40.4469,  -80.0057, "America/New_York"),
    "SD":  (32.7073, -117.1566, "America/Los_Angeles"),
    "SF":  (37.7786, -122.3893, "America/Los_Angeles"),
    "SEA": (47.5914, -122.3325, "America/Los_Angeles"),
    "STL": (38.6226,  -90.1928, "America/Chicago"),
    "TB":  (27.7682,  -82.6534, "America/New_York"),
    "TEX": (32.7473,  -97.0845, "America/Chicago"),
    "TOR": (43.6414,  -79.3894, "America/Toronto"),
    "WSH": (38.8730,  -77.0074, "America/New_York"),
}

# Retractable roof or dome stadiums — weather conditions do not affect play
ROOF_STADIUMS: set[str] = {"AZ", "HOU", "MIA", "MIL", "SEA", "TB", "TEX", "TOR"}

# Approximate compass bearing from home plate to center field (degrees from North).
# Determines how much wind is "blowing out" vs "blowing in."
PARK_CF_BEARING: dict[str, float] = {
    "ATL": 10,  "BAL": 62,  "BOS": 107, "CHC": 170, "CIN": 90,
    "CLE": 25,  "COL": 285, "CWS": 5,   "DET": 270, "KC":  5,
    "LAA": 270, "LAD": 330, "MIN": 340, "NYM": 45,  "NYY": 5,
    "ATH": 310, "PHI": 50,  "PIT": 5,   "SD":  5,   "SF":  112,
    "STL": 5,   "WSH": 5,
}

LEAGUE_AVG_WEATHER = {
    "wind_speed_mph":     7.0,
    "wind_direction_deg": 180.0,
    "wind_component_out": 0.0,
    "temperature_f":      68.0,
    "precip_probability": 0.05,
}


def _wind_component_out(wind_speed_mph: float, wind_dir_deg: float, home_team: str) -> float:
    """
    Project wind speed onto the home-plate → center-field axis.
    Positive = blowing out (hitter-friendly), negative = blowing in.
    Returns 0 for dome/retractable-roof parks or parks without bearing data.
    """
    if home_team in ROOF_STADIUMS:
        return 0.0
    bearing = PARK_CF_BEARING.get(home_team)
    if bearing is None:
        return 0.0
    # Wind blows FROM wind_dir_deg; component toward center field:
    angle_diff = math.radians(wind_dir_deg - bearing)
    return round(wind_speed_mph * math.cos(angle_diff), 2)


def _parse_hourly(data: dict, target_hour: int) -> Optional[dict]:
    """Extract weather values for the first time-slot at or after target_hour (local)."""
    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return None

    idx = None
    for i, t in enumerate(times):
        try:
            hour = int(t[11:13])
        except (IndexError, ValueError):
            continue
        if hour >= target_hour:
            idx = i
            break

    if idx is None:
        idx = -1  # fallback: last available hour

    try:
        return {
            "wind_speed_mph":     float(hourly["windspeed_10m"][idx]),
            "wind_direction_deg": float(hourly["winddirection_10m"][idx]),
            "temperature_f":      float(hourly["temperature_2m"][idx]),
            "precip_probability": float(hourly["precipitation_probability"][idx]) / 100.0,
        }
    except (KeyError, IndexError, TypeError):
        return None


def get_game_weather(home_team: str, game_hour_local: int = 19) -> dict:
    """
    Return weather features for today's game at home_team's stadium.
    game_hour_local: approximate local start hour (default 19 = 7pm).
    Falls back to LEAGUE_AVG_WEATHER on any failure.
    """
    if home_team in ROOF_STADIUMS:
        return {**LEAGUE_AVG_WEATHER, "wind_speed_mph": 0.0, "wind_component_out": 0.0,
                "temperature_f": 72.0, "precip_probability": 0.0}

    if home_team not in STADIUM_COORDS:
        return LEAGUE_AVG_WEATHER.copy()

    lat, lon, tz = STADIUM_COORDS[home_team]
    try:
        resp = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude":            lat,
                "longitude":           lon,
                "hourly":              "windspeed_10m,winddirection_10m,temperature_2m,precipitation_probability",
                "wind_speed_unit":     "mph",
                "temperature_unit":    "fahrenheit",
                "forecast_days":       2,
                "timezone":            tz,
            },
            timeout=8,
        )
        resp.raise_for_status()
        parsed = _parse_hourly(resp.json(), game_hour_local)
        if parsed is None:
            return LEAGUE_AVG_WEATHER.copy()

        return {
            **parsed,
            "wind_component_out": _wind_component_out(
                parsed["wind_speed_mph"], parsed["wind_direction_deg"], home_team
            ),
        }
    except Exception as e:
        print(f"  Warning: weather fetch failed for {home_team} — {e}")
        return LEAGUE_AVG_WEATHER.copy()


def get_historical_weather(home_team: str, game_date: str, game_hour_local: int = 19) -> dict:
    """
    Fetch historical weather from the Open-Meteo archive API.
    game_date: "YYYY-MM-DD"
    Used by backfill_weather.py to populate the weather_cache table.
    """
    if home_team in ROOF_STADIUMS:
        return {**LEAGUE_AVG_WEATHER, "wind_speed_mph": 0.0, "wind_component_out": 0.0,
                "temperature_f": 72.0, "precip_probability": 0.0}

    if home_team not in STADIUM_COORDS:
        return LEAGUE_AVG_WEATHER.copy()

    lat, lon, tz = STADIUM_COORDS[home_team]
    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":         lat,
                "longitude":        lon,
                "start_date":       game_date,
                "end_date":         game_date,
                "hourly":           "windspeed_10m,winddirection_10m,temperature_2m,precipitation",
                "wind_speed_unit":  "mph",
                "temperature_unit": "fahrenheit",
                "timezone":         tz,
            },
            timeout=12,
        )
        resp.raise_for_status()
        data   = resp.json()
        hourly = data.get("hourly", {})

        # Archive API returns precipitation, not precipitation_probability.
        # Use precip > 0.1mm as proxy for rain.
        times = hourly.get("time", [])
        idx   = None
        for i, t in enumerate(times):
            try:
                if int(t[11:13]) >= game_hour_local:
                    idx = i
                    break
            except (IndexError, ValueError):
                continue
        if idx is None:
            idx = min(game_hour_local, len(times) - 1)

        wind_speed  = float(hourly["windspeed_10m"][idx])
        wind_dir    = float(hourly["winddirection_10m"][idx])
        temp_f      = float(hourly["temperature_2m"][idx])
        precip_mm   = float(hourly.get("precipitation", [0.0])[idx])
        precip_prob = min(precip_mm / 2.0, 1.0)  # rough: 2mm = ~100% rain

        return {
            "wind_speed_mph":     round(wind_speed, 1),
            "wind_direction_deg": round(wind_dir, 1),
            "wind_component_out": _wind_component_out(wind_speed, wind_dir, home_team),
            "temperature_f":      round(temp_f, 1),
            "precip_probability": round(precip_prob, 3),
        }
    except Exception as e:
        print(f"  Warning: historical weather failed for {home_team} on {game_date} — {e}")
        return LEAGUE_AVG_WEATHER.copy()
