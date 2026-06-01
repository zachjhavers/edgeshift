"""
Fetch NHL game schedules and final scores from the public NHL API.

Incremental mode (default): fetches from the last recorded game date forward.
Full mode (--full):          iterates through all defined season date ranges.

NHL API base: https://api-web.nhle.com
  GET /v1/schedule/{date}  → Returns a week of games starting from {date}.
"""

import sys
import time
from datetime import datetime, timedelta

import requests

from db import get_conn, setup_db
from utils import NHL_API_BASE, SEASONS


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "edgeshift-nhl-engine/1.0"})


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)
    return {}


def _result_from_game(game: dict) -> str:
    """Derive HOME_WIN / AWAY_WIN / TBD from a schedule game object."""
    state = game.get("gameState", "")
    if state not in ("OFF", "FINAL"):
        return "TBD"
    home_score = (game.get("homeTeam") or {}).get("score")
    away_score = (game.get("awayTeam") or {}).get("score")
    if home_score is None or away_score is None:
        return "TBD"
    return "HOME_WIN" if home_score > away_score else "AWAY_WIN"


def _upsert_games(games: list[dict]):
    if not games:
        return
    conn = get_conn()
    conn.executemany(
        """INSERT INTO games
               (game_id, game_date, season, game_type,
                home_team, away_team, home_score, away_score, result)
           VALUES (?,?,?,?,?,?,?,?,?)
           ON CONFLICT(game_id) DO UPDATE SET
               home_score = excluded.home_score,
               away_score = excluded.away_score,
               result     = excluded.result
        """,
        games,
    )
    conn.commit()
    conn.close()


def _parse_week(data: dict) -> list[dict]:
    rows = []
    for day in data.get("gameWeek", []):
        date = day.get("date", "")
        for g in day.get("games", []):
            game_id   = str(g.get("id", ""))
            game_type = int(g.get("gameType", 0))
            season    = str(g.get("season", ""))
            home      = (g.get("homeTeam") or {}).get("abbrev", "")
            away      = (g.get("awayTeam") or {}).get("abbrev", "")
            if not game_id or not home or not away:
                continue
            # Only regular season (2) and playoff (3) games
            if game_type not in (2, 3):
                continue
            home_score = (g.get("homeTeam") or {}).get("score")
            away_score = (g.get("awayTeam") or {}).get("score")
            result     = _result_from_game(g)
            rows.append((game_id, date, season, game_type,
                         home, away, home_score, away_score, result))
    return rows


def _last_fetched_date() -> str | None:
    conn = get_conn()
    row = conn.execute("SELECT MAX(game_date) FROM games").fetchone()
    conn.close()
    return row[0] if row else None


def fetch_schedule_range(start_date: str, end_date: str):
    """Fetch all games in [start_date, end_date] by stepping one week at a time."""
    current = datetime.strptime(start_date, "%Y-%m-%d")
    end     = datetime.strptime(end_date,   "%Y-%m-%d")
    total_inserted = 0

    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        url  = f"{NHL_API_BASE}/v1/schedule/{date_str}"
        try:
            data = _get(url)
        except Exception as e:
            print(f"  Warning: failed to fetch {date_str}: {e}")
            current += timedelta(days=7)
            continue

        rows = _parse_week(data)
        _upsert_games(rows)
        total_inserted += len(rows)
        print(f"  {date_str}: {len(rows)} games")

        # The response covers up to 7 days; jump forward by 7
        last_day_in_response = max(
            (day.get("date", date_str) for day in data.get("gameWeek", [])),
            default=date_str,
        )
        try:
            current = datetime.strptime(last_day_in_response, "%Y-%m-%d") + timedelta(days=1)
        except ValueError:
            current += timedelta(days=7)

        time.sleep(0.2)  # polite rate limiting

    return total_inserted


def fetch_schedule(full: bool = False):
    """
    Fetch schedule incrementally or in full.

    full=True:  iterate all seasons defined in SEASONS.
    full=False: fetch from (last recorded date - 3 days) to today + 2 days.
    """
    if full:
        print("Fetching full schedule history...")
        for season, start, end in SEASONS:
            print(f"\n  Season {season}: {start} → {end}")
            n = fetch_schedule_range(start, end)
            print(f"  Season {season}: {n} game rows stored.")
    else:
        last = _last_fetched_date()
        if last:
            start = (datetime.strptime(last, "%Y-%m-%d") - timedelta(days=3)).strftime("%Y-%m-%d")
        else:
            # No data — fall back to current season start
            start = SEASONS[-1][1]

        end = (datetime.now() + timedelta(days=2)).strftime("%Y-%m-%d")
        print(f"Fetching schedule: {start} → {end}")
        n = fetch_schedule_range(start, end)
        print(f"Done: {n} game rows stored.")


if __name__ == "__main__":
    setup_db()
    fetch_schedule(full="--full" in sys.argv)
