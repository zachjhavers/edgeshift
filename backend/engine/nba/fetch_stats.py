"""
Fetch NBA game box scores from the ESPN unofficial API.
No authentication required. Works from cloud servers.

Backfill strategy:
  1. Iterate through each date in NBA seasons (skip off-season months)
  2. GET scoreboard?dates=YYYYMMDD  → collect completed game IDs
  3. GET summary?event=ID           → parse box scores per game
  4. Upsert into games table

ESPN stat keys used:
  fieldGoalsMade-fieldGoalsAttempted
  threePointFieldGoalsMade-threePointFieldGoalsAttempted
  freeThrowsMade-freeThrowsAttempted
  offensiveRebounds, defensiveRebounds, turnovers
"""

import time
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

from db import get_conn, setup_db

ESPN_BASE     = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
REQUEST_DELAY = 0.35   # seconds between API calls

# ESPN uses non-standard abbreviations for some teams
ESPN_ABBR = {
    "NY":   "NYK",
    "GS":   "GSW",
    "SA":   "SAS",
    "NO":   "NOP",
    "WSH":  "WAS",
    "UTAH": "UTA",
    "PHX":  "PHX",
}


def _norm(abbr: str) -> str:
    return ESPN_ABBR.get(abbr.upper(), abbr.upper())


def _get(path: str, params: dict | None = None, retries: int = 2) -> dict | None:
    url = f"{ESPN_BASE}/{path}"
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=12)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2)
    return None


def _parse_stat(stats: dict, key: str) -> int:
    """Parse 'made-attempted' or plain integer stat."""
    val = stats.get(key, "0")
    if "-" in str(val):
        return int(str(val).split("-")[0])
    try:
        return int(float(str(val)))
    except (ValueError, TypeError):
        return 0


def _fetch_game_ids_for_date(game_date: str) -> list[tuple[str, int]]:
    """
    GET scoreboard for a single date.
    Returns list of (game_id, season_type) for completed games only.
    game_date: YYYYMMDD
    """
    data = _get("scoreboard", {"dates": game_date})
    if not data:
        return []

    results = []
    for event in data.get("events", []):
        comp   = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {})

        # Only completed games (statusId 3 = final)
        if status.get("completed") is not True:
            continue

        season_type = event.get("season", {}).get("type", 2)
        results.append((event["id"], season_type))

    return results


def _fetch_box_score(game_id: str) -> dict | None:
    """
    GET summary for one game. Returns structured dict or None.
    """
    data = _get("summary", {"event": game_id})
    if not data:
        return None

    bs = data.get("boxscore", {})
    teams = bs.get("teams", [])
    if len(teams) < 2:
        return None

    header      = data.get("header", {})
    season_info = header.get("season", {})
    season_year = season_info.get("year", 0)
    season_type = season_info.get("type", 2)
    game_date   = header.get("competitions", [{}])[0].get("date", "")[:10]
    season      = f"{season_year-1}-{str(season_year)[2:]}" if season_year else "unknown"
    game_type   = "playoff" if season_type == 3 else "regular"

    # competitions[0].competitors: homeAway + score
    comps = header.get("competitions", [{}])[0].get("competitors", [])
    scores: dict[str, int] = {}
    for c in comps:
        abbr = _norm(c.get("team", {}).get("abbreviation", ""))
        try:
            scores[abbr] = int(float(c.get("score", 0)))
        except (ValueError, TypeError):
            scores[abbr] = 0

    home_abbr = away_abbr = None
    home_stats = away_stats = {}

    for team_data in teams:
        abbr    = _norm(team_data["team"]["abbreviation"])
        ha      = team_data.get("homeAway", "")
        raw     = {s["name"]: s.get("displayValue", "0") for s in team_data.get("statistics", [])}

        def _s(key):
            return _parse_stat(raw, key)

        parsed = {
            "pts":  scores.get(abbr, 0),
            "fgm":  _s("fieldGoalsMade-fieldGoalsAttempted"),
            "fga":  _parse_stat({"v": raw.get("fieldGoalsMade-fieldGoalsAttempted", "0-0").split("-")[-1]}, "v")
                    if "-" in str(raw.get("fieldGoalsMade-fieldGoalsAttempted", "")) else 0,
            "fg3m": _s("threePointFieldGoalsMade-threePointFieldGoalsAttempted"),
            "fg3a": int(str(raw.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "0-0")).split("-")[-1])
                    if "-" in str(raw.get("threePointFieldGoalsMade-threePointFieldGoalsAttempted", "")) else 0,
            "ftm":  _s("freeThrowsMade-freeThrowsAttempted"),
            "fta":  int(str(raw.get("freeThrowsMade-freeThrowsAttempted", "0-0")).split("-")[-1])
                    if "-" in str(raw.get("freeThrowsMade-freeThrowsAttempted", "")) else 0,
            "oreb": _s("offensiveRebounds"),
            "dreb": _s("defensiveRebounds"),
            "tov":  _s("turnovers"),
        }

        if ha == "home":
            home_abbr  = abbr
            home_stats = parsed
        else:
            away_abbr  = abbr
            away_stats = parsed

    if not home_abbr or not away_abbr or not home_stats or not away_stats:
        return None

    home_win = 1 if home_stats["pts"] > away_stats["pts"] else 0

    return {
        "game_id":   game_id,
        "game_date": game_date,
        "season":    season,
        "game_type": game_type,
        "home_team": home_abbr,
        "away_team": away_abbr,
        "home_win":  home_win,
        "home":      home_stats,
        "away":      away_stats,
    }


def _store_game(g: dict) -> bool:
    conn = get_conn()
    h, a = g["home"], g["away"]
    try:
        conn.execute("""
            INSERT INTO games (
                game_id, game_date, season, game_type, home_team, away_team,
                home_pts, home_fgm, home_fga, home_fg3m, home_fg3a,
                home_ftm, home_fta, home_oreb, home_dreb, home_tov,
                away_pts, away_fgm, away_fga, away_fg3m, away_fg3a,
                away_ftm, away_fta, away_oreb, away_dreb, away_tov,
                home_win
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(game_id) DO UPDATE SET
                home_pts = excluded.home_pts,
                away_pts = excluded.away_pts,
                home_win = excluded.home_win
        """, (
            g["game_id"], g["game_date"], g["season"], g["game_type"],
            g["home_team"], g["away_team"],
            h["pts"], h["fgm"], h["fga"], h["fg3m"], h["fg3a"],
            h["ftm"], h["fta"], h["oreb"], h["dreb"], h["tov"],
            a["pts"], a["fgm"], a["fga"], a["fg3m"], a["fg3a"],
            a["ftm"], a["fta"], a["oreb"], a["dreb"], a["tov"],
            g["home_win"],
        ))
        conn.commit()
        conn.close()
        return True
    except Exception:
        conn.close()
        return False


def _nba_dates(start_year: int, end_year: int):
    """
    Yield every date in NBA season windows (Oct–Jun), skipping off-season.
    start_year / end_year are the season-ending years (e.g. 2016 for 2015-16).
    """
    for year in range(start_year, end_year + 1):
        # Regular season: late Oct to mid-April
        # Playoffs: mid-April to mid-June
        season_start = date(year - 1, 10, 1)
        season_end   = date(year, 6, 30)
        d = season_start
        while d <= season_end:
            yield d
            d += timedelta(days=1)


def backfill(first_year: int = 2016):
    """
    Backfill all NBA games from first_year season through current.
    first_year = season ending year (2016 = 2015-16 season).
    Estimated time: ~60-90 minutes for 10 seasons.
    """
    setup_db()
    today      = date.today()
    end_year   = today.year if today.month >= 10 else today.year
    total_games = 0
    skipped     = 0

    # Check which game IDs are already stored
    conn = get_conn()
    existing = set(r[0] for r in conn.execute("SELECT game_id FROM games").fetchall())
    conn.close()
    print(f"Already have {len(existing)} games. Fetching new ones...")

    for d in _nba_dates(first_year, end_year):
        if d > today:
            break

        date_str = d.strftime("%Y%m%d")
        game_ids = _fetch_game_ids_for_date(date_str)
        time.sleep(REQUEST_DELAY)

        new_ids = [(gid, stype) for gid, stype in game_ids if gid not in existing]
        if not new_ids:
            continue

        for game_id, _ in new_ids:
            g = _fetch_box_score(game_id)
            time.sleep(REQUEST_DELAY)

            if g is None:
                skipped += 1
                continue

            if _store_game(g):
                existing.add(game_id)
                total_games += 1

        if total_games % 100 == 0 and total_games > 0:
            print(f"  {d}  |  {total_games} games stored so far...")

    print(f"Backfill complete — {total_games} games stored, {skipped} skipped.")


def fetch_recent(days_back: int = 5):
    """Fetch the last N days of completed games to keep DB current."""
    setup_db()
    today = date.today()
    count = 0
    for i in range(days_back, -1, -1):
        d        = today - timedelta(days=i)
        date_str = d.strftime("%Y%m%d")
        game_ids = _fetch_game_ids_for_date(date_str)
        time.sleep(REQUEST_DELAY)

        for game_id, _ in game_ids:
            g = _fetch_box_score(game_id)
            time.sleep(REQUEST_DELAY)
            if g and _store_game(g):
                count += 1

    print(f"  {count} recent game(s) updated.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    backfill()
