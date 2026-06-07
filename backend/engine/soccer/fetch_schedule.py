"""
Fetch upcoming FIFA World Cup 2026 fixtures from the ESPN unofficial API.
Stores upcoming matches (not yet played) into the matches table with result=TBD.
"""

import time
from datetime import datetime, timedelta

import requests

from db import get_conn
from utils import canonical

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WC_SLUG   = "fifa.world"     # ESPN league slug for FIFA World Cup


def _norm(name: str) -> str:
    return canonical(name)


def get_upcoming_matches(days_ahead: int = 7) -> list[dict]:
    """Return scheduled WC matches for the next `days_ahead` days."""
    today = datetime.now()
    matches = []

    for offset in range(days_ahead + 1):
        date = (today + timedelta(days=offset)).strftime("%Y%m%d")
        try:
            r = requests.get(
                f"{ESPN_BASE}/{WC_SLUG}/scoreboard",
                params={"dates": date},
                timeout=12,
            )
            r.raise_for_status()
            data = r.json()
            time.sleep(0.3)
        except Exception as e:
            print(f"  ESPN schedule fetch failed for {date}: {e}")
            continue

        for event in data.get("events", []):
            comp        = event.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            status      = comp.get("status", {}).get("type", {}).get("state", "pre")

            if status == "post":
                continue  # already played

            home_name = away_name = None
            for c in competitors:
                name = _norm(c.get("team", {}).get("displayName", ""))
                if c.get("homeAway") == "home":
                    home_name = name
                else:
                    away_name = name

            if not home_name or not away_name:
                continue

            # WC is neutral venue — label both sides by FIFA convention
            # ESPN sets "home" to the first-listed team; we treat all as neutral
            match_date = date[:4] + "-" + date[4:6] + "-" + date[6:]
            matches.append({
                "match_date": match_date,
                "home_team":  home_name,
                "away_team":  away_name,
                "tournament": "FIFA World Cup",
                "neutral":    1,
            })

    return matches


def store_upcoming(matches: list[dict]) -> int:
    conn = get_conn()
    stored = 0
    for m in matches:
        try:
            conn.execute("""
                INSERT INTO matches
                    (match_date, home_team, away_team, tournament, neutral, result)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(match_date, home_team, away_team) DO NOTHING
            """, (m["match_date"], m["home_team"], m["away_team"],
                  m["tournament"], m["neutral"], "TBD"))
            stored += 1
        except Exception:
            pass
    conn.commit()
    conn.close()
    return stored


def fetch_and_store_schedule(days_ahead: int = 7) -> int:
    matches = get_upcoming_matches(days_ahead)
    if not matches:
        print("  No upcoming WC matches found.")
        return 0
    n = store_upcoming(matches)
    print(f"  Stored {n} upcoming match(es).")
    return n


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    for m in get_upcoming_matches(14):
        print(m)
