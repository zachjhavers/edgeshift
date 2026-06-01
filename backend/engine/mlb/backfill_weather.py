"""
One-time backfill: fetch historical weather for all games in the DB and
populate the weather_cache table.  Run once before retraining the model.

  python backfill_weather.py                     # all dates in statcast_raw
  python backfill_weather.py --start 2023-01-01  # from a specific date
  python backfill_weather.py --force              # re-fetch already-cached rows

Rate limit: Open-Meteo archive allows up to 10,000 requests/day on the free tier.
With 30 teams and ~180 game-days per season, a full backfill over 3 seasons is
roughly 30 * 540 = ~16,000 per-stadium-day calls.  The script batches by year
(one API call per stadium-year = ~90 calls total) to stay well within limits.
"""

import argparse
import time
from datetime import date, timedelta

import pandas as pd
import requests
from sqlalchemy import text

from db import get_engine
from weather import STADIUM_COORDS, ROOF_STADIUMS, _wind_component_out, LEAGUE_AVG_WEATHER


def _ensure_weather_cache(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS weather_cache (
                game_date           DATE        NOT NULL,
                home_team           VARCHAR(10) NOT NULL,
                wind_speed_mph      FLOAT       NOT NULL,
                wind_direction_deg  FLOAT       NOT NULL,
                wind_component_out  FLOAT       NOT NULL,
                temperature_f       FLOAT       NOT NULL,
                precip_probability  FLOAT       NOT NULL,
                PRIMARY KEY (game_date, home_team)
            )
        """))


def _fetch_year_weather(home_team: str, year: int, game_dates: list[str]) -> dict[str, dict]:
    """
    Fetch a full year of hourly weather for one stadium in one API call.
    Returns {game_date_str: weather_dict}.
    """
    if home_team not in STADIUM_COORDS:
        return {}

    lat, lon, tz = STADIUM_COORDS[home_team]
    start = f"{year}-01-01"
    end   = f"{year}-12-31"

    try:
        resp = requests.get(
            "https://archive-api.open-meteo.com/v1/archive",
            params={
                "latitude":         lat,
                "longitude":        lon,
                "start_date":       start,
                "end_date":         end,
                "hourly":           "windspeed_10m,winddirection_10m,temperature_2m,precipitation",
                "wind_speed_unit":  "mph",
                "temperature_unit": "fahrenheit",
                "timezone":         tz,
            },
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"    Weather API error for {home_team} {year}: {e}")
        return {}

    hourly = resp.json().get("hourly", {})
    times  = hourly.get("time", [])
    if not times:
        return {}

    # Build a lookup: date_str → index of first hour >= 18:00 (6pm local)
    time_idx: dict[str, int] = {}
    for i, t in enumerate(times):
        d = t[:10]
        h = int(t[11:13])
        if d not in time_idx and h >= 18:
            time_idx[d] = i

    results: dict[str, dict] = {}
    game_date_set = set(game_dates)

    for d, idx in time_idx.items():
        if d not in game_date_set:
            continue
        try:
            wind_speed  = float(hourly["windspeed_10m"][idx])
            wind_dir    = float(hourly["winddirection_10m"][idx])
            temp_f      = float(hourly["temperature_2m"][idx])
            precip_mm   = float(hourly.get("precipitation", [0.0] * (idx + 1))[idx])
            precip_prob = min(precip_mm / 2.0, 1.0)

            if home_team in ROOF_STADIUMS:
                results[d] = {**LEAGUE_AVG_WEATHER,
                              "wind_speed_mph": 0.0, "wind_component_out": 0.0,
                              "temperature_f": 72.0, "precip_probability": 0.0}
            else:
                results[d] = {
                    "wind_speed_mph":     round(wind_speed, 1),
                    "wind_direction_deg": round(wind_dir, 1),
                    "wind_component_out": _wind_component_out(wind_speed, wind_dir, home_team),
                    "temperature_f":      round(temp_f, 1),
                    "precip_probability": round(precip_prob, 3),
                }
        except (IndexError, KeyError, TypeError):
            continue

    return results


def backfill(start_date: str | None = None, force: bool = False) -> None:
    engine = get_engine()
    _ensure_weather_cache(engine)

    # Load distinct (game_date, home_team) pairs from statcast_raw
    query = "SELECT DISTINCT CAST(game_date AS DATE)::text AS gd, home_team FROM statcast_raw WHERE game_type = 'R'"
    if start_date:
        query += f" AND CAST(game_date AS DATE) >= '{start_date}'"
    df_games = pd.read_sql(query, engine)
    if df_games.empty:
        print("No games found in statcast_raw.")
        return

    if not force:
        cached = pd.read_sql(
            "SELECT game_date::text AS gd, home_team FROM weather_cache", engine
        )
        cached_keys = set(zip(cached["gd"], cached["home_team"]))
        before = len(df_games)
        df_games = df_games[
            ~df_games.apply(lambda r: (r["gd"], r["home_team"]) in cached_keys, axis=1)
        ]
        print(f"Skipping {before - len(df_games)} already-cached rows.")

    if df_games.empty:
        print("All rows already cached. Use --force to re-fetch.")
        return

    # Group by (home_team, year) for batch API calls
    df_games["year"] = df_games["gd"].str[:4].astype(int)
    groups = df_games.groupby(["home_team", "year"])
    total  = len(groups)

    print(f"Backfilling weather for {len(df_games)} game-days across {total} stadium-year groups...")

    inserted = 0
    for (team, year), group in groups:
        dates = group["gd"].tolist()
        print(f"  {team} {year} ({len(dates)} games)...", end=" ", flush=True)
        weather_map = _fetch_year_weather(team, year, dates)

        rows = []
        for d, w in weather_map.items():
            rows.append({
                "game_date":          d,
                "home_team":          team,
                "wind_speed_mph":     w["wind_speed_mph"],
                "wind_direction_deg": w["wind_direction_deg"],
                "wind_component_out": w["wind_component_out"],
                "temperature_f":      w["temperature_f"],
                "precip_probability": w["precip_probability"],
            })

        if rows:
            with engine.begin() as conn:
                for r in rows:
                    conn.execute(text("""
                        INSERT INTO weather_cache
                            (game_date, home_team, wind_speed_mph, wind_direction_deg,
                             wind_component_out, temperature_f, precip_probability)
                        VALUES
                            (:game_date, :home_team, :wind_speed_mph, :wind_direction_deg,
                             :wind_component_out, :temperature_f, :precip_probability)
                        ON CONFLICT (game_date, home_team) DO UPDATE SET
                            wind_speed_mph      = EXCLUDED.wind_speed_mph,
                            wind_direction_deg  = EXCLUDED.wind_direction_deg,
                            wind_component_out  = EXCLUDED.wind_component_out,
                            temperature_f       = EXCLUDED.temperature_f,
                            precip_probability  = EXCLUDED.precip_probability
                    """), r)
            inserted += len(rows)
            print(f"{len(rows)} inserted.")
        else:
            print("no data.")

        time.sleep(0.1)  # gentle rate limiting

    print(f"\nDone. {inserted} rows written to weather_cache.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill historical weather into weather_cache")
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD")
    parser.add_argument("--force", action="store_true", help="Re-fetch already-cached rows")
    args = parser.parse_args()
    backfill(start_date=args.start, force=args.force)
