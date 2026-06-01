"""
Fetch per-game team and goalie stats from the NHL API boxscore endpoint.

For each game in the DB that has no team_game_stats yet, fetches:
  - Goals, shots on goal, PP goals/opp, PK goals against/opp  (per team)
  - Saves, shots against, decision                             (per goalie)

NHL API: GET https://api-web.nhle.com/v1/gamecenter/{gameId}/boxscore
"""

import time

import requests

from db import get_conn
from utils import NHL_API_BASE


_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "edgeshift-nhl-engine/1.0"})


def _get(url: str, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = _SESSION.get(url, timeout=15)
            if resp.status_code == 404:
                return {}
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt == retries - 1:
                return {}
            time.sleep(2 ** attempt)
    return {}



def _parse_boxscore(game_id: str, data: dict) -> tuple[list, list]:
    """
    Returns (team_rows, goalie_rows) ready for DB insert.
    team_rows:   [(game_id, team_code, goals, shots, pp_goals, pp_opp, pk_ga, pk_opp)]
    goalie_rows: [(game_id, goalie_id, team_code, shots_against, saves, decision)]
    """
    if not data:
        return [], []

    home_abbrev = (data.get("homeTeam") or {}).get("abbrev", "")
    away_abbrev = (data.get("awayTeam") or {}).get("abbrev", "")
    home_score  = (data.get("homeTeam") or {}).get("score", 0) or 0
    away_score  = (data.get("awayTeam") or {}).get("score", 0) or 0

    summary = data.get("summary") or {}
    stats_list = summary.get("teamGameStats") or []
    stats_map  = {s.get("category"): s for s in stats_list}

    def _val(category: str, side: str):
        entry = stats_map.get(category, {})
        return entry.get(f"{side}Value", 0) if entry else 0

    def _to_int(v) -> int:
        try:
            return int(v)
        except (TypeError, ValueError):
            return 0

    home_shots   = _to_int(_val("sog", "home"))
    away_shots   = _to_int(_val("sog", "away"))
    home_pp_g    = _to_int(_val("powerPlayGoals", "home"))
    away_pp_g    = _to_int(_val("powerPlayGoals", "away"))
    home_pp_opp  = _to_int(_val("powerPlayOpportunities", "home"))
    away_pp_opp  = _to_int(_val("powerPlayOpportunities", "away"))

    # PK for a team = opponent's PP against them
    # home_pk_goals_against = away PP goals scored; home_pk_opp = away PP opps
    team_rows = [
        (game_id, home_abbrev, home_score, home_shots,
         home_pp_g, home_pp_opp, away_pp_g, away_pp_opp),
        (game_id, away_abbrev, away_score, away_shots,
         away_pp_g, away_pp_opp, home_pp_g, home_pp_opp),
    ]
    team_rows = [(r[0], r[1]) + r[2:] for r in team_rows if r[1]]  # skip blank abbrevs

    # Goalie stats
    goalie_rows = []
    player_stats = data.get("playerByGameStats") or {}
    for side_key, team_code in [("homeTeam", home_abbrev), ("awayTeam", away_abbrev)]:
        goalies = (player_stats.get(side_key) or {}).get("goalies") or []
        for g in goalies:
            goalie_id    = str(g.get("playerId", ""))
            shots_ag     = _to_int(g.get("shotsAgainst", 0))
            saves        = _to_int(g.get("saves", 0))
            decision     = str(g.get("decision") or "")
            if goalie_id and shots_ag > 0:
                goalie_rows.append((game_id, goalie_id, team_code, shots_ag, saves, decision))

    return team_rows, goalie_rows


def _games_missing_stats() -> list[str]:
    conn = get_conn()
    rows = conn.execute("""
        SELECT g.game_id FROM games g
        LEFT JOIN team_game_stats t ON t.game_id = g.game_id
        WHERE t.game_id IS NULL
          AND g.result IN ('HOME_WIN', 'AWAY_WIN')
        ORDER BY g.game_date
    """).fetchall()
    conn.close()
    return [str(r[0]) for r in rows]


def fetch_stats():
    missing = _games_missing_stats()
    if not missing:
        print("All game stats already fetched.")
        return

    print(f"Fetching boxscores for {len(missing)} games...")
    inserted_teams   = 0
    inserted_goalies = 0

    conn = get_conn()
    for i, game_id in enumerate(missing, 1):
        url  = f"{NHL_API_BASE}/v1/gamecenter/{game_id}/boxscore"
        data = _get(url)
        if not data:
            continue

        team_rows, goalie_rows = _parse_boxscore(game_id, data)

        if team_rows:
            conn.executemany(
                """INSERT OR REPLACE INTO team_game_stats
                       (game_id, team_code, goals, shots,
                        pp_goals, pp_opp, pk_goals_against, pk_opp)
                   VALUES (?,?,?,?,?,?,?,?)""",
                team_rows,
            )
            inserted_teams += len(team_rows)

        if goalie_rows:
            conn.executemany(
                """INSERT OR REPLACE INTO goalie_game_stats
                       (game_id, goalie_id, team_code, shots_against, saves, decision)
                   VALUES (?,?,?,?,?,?)""",
                goalie_rows,
            )
            inserted_goalies += len(goalie_rows)

        if i % 100 == 0:
            conn.commit()
            print(f"  {i}/{len(missing)} games processed...")

        time.sleep(0.15)

    conn.commit()
    conn.close()
    print(f"Done: {inserted_teams} team stat rows, {inserted_goalies} goalie stat rows.")


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    fetch_stats()
