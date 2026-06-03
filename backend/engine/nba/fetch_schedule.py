"""
Fetch today's NBA schedule from the ESPN unofficial API.
Returns upcoming AND in-progress games (not just completed).
"""

import time
from datetime import datetime
from pathlib import Path

import requests

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"

ESPN_ABBR = {
    "NY":   "NYK",
    "GS":   "GSW",
    "SA":   "SAS",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
}


def _norm(abbr: str) -> str:
    return ESPN_ABBR.get(abbr.upper(), abbr.upper())


def get_today_games(date: str | None = None) -> list[dict]:
    """
    Return all NBA games scheduled for a given date (or today).
    date: ISO YYYY-MM-DD or None for today.
    Includes scheduled, in-progress, and completed games.
    """
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")

    date_espn = date.replace("-", "")

    try:
        r = requests.get(
            f"{ESPN_BASE}/scoreboard",
            params={"dates": date_espn},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        time.sleep(0.3)
    except Exception as e:
        print(f"  ESPN schedule fetch failed: {e}")
        return []

    games = []
    for event in data.get("events", []):
        comp        = event.get("competitions", [{}])[0]
        competitors = comp.get("competitors", [])
        season_type = event.get("season", {}).get("type", 2)
        game_type   = "playoff" if season_type == 3 else "regular"

        home_abbr = away_abbr = None
        for c in competitors:
            abbr = _norm(c.get("team", {}).get("abbreviation", ""))
            if c.get("homeAway") == "home":
                home_abbr = abbr
            else:
                away_abbr = abbr

        if not home_abbr or not away_abbr:
            continue

        games.append({
            "game_id":   event["id"],
            "game_date": date,
            "home_team": home_abbr,
            "away_team": away_abbr,
            "game_type": game_type,
        })

    return games


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    for g in get_today_games():
        print(g)
