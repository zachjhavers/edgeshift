"""
Fetch today's NBA schedule via nba_api Scoreboard endpoint.
Detects game type (regular/playoff) from the NBA game ID format:
  game_id[2] == '2' → Regular Season
  game_id[2] == '4' → Playoffs
"""

import time
from datetime import datetime
from pathlib import Path


def _game_type(game_id: str) -> str:
    return "playoff" if len(game_id) >= 3 and game_id[2] == "4" else "regular"


def get_today_games(date: str | None = None) -> list[dict]:
    """
    Return today's (or given date's) scheduled NBA games.
    date: ISO format YYYY-MM-DD, or None for today.
    Returns list of {game_id, game_date, home_team, away_team, game_type}.
    """
    from nba_api.stats.endpoints import Scoreboard

    if date is None:
        api_date = datetime.now().strftime("%m/%d/%Y")
        iso_date = datetime.now().strftime("%Y-%m-%d")
    else:
        d        = datetime.strptime(date, "%Y-%m-%d")
        api_date = d.strftime("%m/%d/%Y")
        iso_date = date

    try:
        sb   = Scoreboard(game_date=api_date, timeout=20)
        dfs  = sb.get_data_frames()
        time.sleep(0.5)
    except Exception as e:
        print(f"  NBA Scoreboard fetch failed: {e}")
        return []

    if len(dfs) < 2:
        return []

    game_header = dfs[0]
    line_score  = dfs[1]

    if game_header.empty:
        return []

    games = []
    for _, gh in game_header.iterrows():
        game_id  = str(gh.get("GAME_ID", ""))
        home_tid = gh.get("HOME_TEAM_ID")
        away_tid = gh.get("VISITOR_TEAM_ID")

        if not game_id:
            continue

        home_rows = line_score[
            (line_score["GAME_ID"] == game_id) &
            (line_score["TEAM_ID"] == home_tid)
        ]
        away_rows = line_score[
            (line_score["GAME_ID"] == game_id) &
            (line_score["TEAM_ID"] == away_tid)
        ]

        if home_rows.empty or away_rows.empty:
            continue

        games.append({
            "game_id":   game_id,
            "game_date": iso_date,
            "home_team": str(home_rows.iloc[0]["TEAM_ABBREVIATION"]),
            "away_team": str(away_rows.iloc[0]["TEAM_ABBREVIATION"]),
            "game_type": _game_type(game_id),
        })

    return games


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    for g in get_today_games():
        print(g)
