"""
Backfill and refresh NBA team game logs from stats.nba.com via nba_api.
Fetches Regular Season and Playoff games for seasons 2015-16 onwards.
Each game is stored as one row with both teams' box score stats joined.
"""

import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from db import get_conn, setup_db

NBA_HEADERS = {
    "Host":                  "stats.nba.com",
    "User-Agent":            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept":                "application/json, text/plain, */*",
    "Accept-Language":       "en-US,en;q=0.9",
    "Accept-Encoding":       "gzip, deflate, br",
    "x-nba-stats-origin":    "stats",
    "x-nba-stats-token":     "true",
    "Referer":               "https://www.nba.com/",
    "Connection":            "keep-alive",
    "Pragma":                "no-cache",
    "Cache-Control":         "no-cache",
}

FIRST_SEASON = "2015-16"


def _current_season() -> str:
    now = datetime.now()
    y   = now.year
    return f"{y}-{str(y+1)[2:]}" if now.month >= 10 else f"{y-1}-{str(y)[2:]}"


def _season_list(first: str, last: str) -> list[str]:
    seasons, y = [], int(first[:4])
    while y <= int(last[:4]):
        seasons.append(f"{y}-{str(y+1)[2:]}")
        y += 1
    return seasons


def _game_type(game_id: str) -> str:
    return "playoff" if len(game_id) >= 3 and game_id[2] == "4" else "regular"


def _fetch_season(season: str, season_type: str) -> pd.DataFrame:
    for attempt in range(2):
        try:
            log = LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                league_id="00",
                direction="ASC",
                sorter="DATE",
                timeout=30,
                headers=NBA_HEADERS,
            )
            df = log.get_data_frames()[0]
            time.sleep(0.8)
            return df
        except Exception as e:
            if attempt == 0:
                print(f"    Fetch failed ({e}) — retrying in 5s.")
                time.sleep(5)
    return pd.DataFrame()


def _store_games(df: pd.DataFrame, season: str) -> int:
    if df.empty:
        return 0

    df = df.copy()
    df["is_home"] = df["MATCHUP"].str.contains(r"vs\.", regex=True).astype(int)

    keep = ["GAME_ID", "GAME_DATE", "TEAM_ABBREVIATION",
            "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
            "OREB", "DREB", "TOV", "is_home"]
    df = df[[c for c in keep if c in df.columns]].copy()

    home = df[df["is_home"] == 1].drop(columns="is_home")
    away = df[df["is_home"] == 0].drop(columns="is_home")

    h = home.rename(columns={
        c: (c if c in ("GAME_ID", "GAME_DATE") else f"home_{c.lower()}")
        for c in home.columns
    })
    a = away.rename(columns={
        c: (c if c == "GAME_ID" else f"away_{c.lower()}")
        for c in away.columns
    })

    merged = pd.merge(h, a, on="GAME_ID", how="inner")
    if merged.empty:
        return 0

    conn  = get_conn()
    count = 0

    def _i(row, col):
        v = row.get(col)
        return int(v) if pd.notna(v) else 0

    for _, r in merged.iterrows():
        game_id   = str(r["GAME_ID"])
        game_date = str(r.get("GAME_DATE", ""))[:10]
        gtype     = _game_type(game_id)
        home_pts  = int(r["home_pts"]) if pd.notna(r.get("home_pts")) else None
        away_pts  = int(r["away_pts"]) if pd.notna(r.get("away_pts")) else None
        home_win  = (1 if home_pts > away_pts else 0) if (home_pts and away_pts) else None

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
                game_id, game_date, season, gtype,
                str(r["home_team_abbreviation"]), str(r["away_team_abbreviation"]),
                home_pts,
                _i(r, "home_fgm"), _i(r, "home_fga"),
                _i(r, "home_fg3m"), _i(r, "home_fg3a"),
                _i(r, "home_ftm"), _i(r, "home_fta"),
                _i(r, "home_oreb"), _i(r, "home_dreb"), _i(r, "home_tov"),
                away_pts,
                _i(r, "away_fgm"), _i(r, "away_fga"),
                _i(r, "away_fg3m"), _i(r, "away_fg3a"),
                _i(r, "away_ftm"), _i(r, "away_fta"),
                _i(r, "away_oreb"), _i(r, "away_dreb"), _i(r, "away_tov"),
                home_win,
            ))
            count += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    return count


def backfill(first_season: str = FIRST_SEASON):
    """Fetch all Regular Season + Playoff data from first_season to present."""
    setup_db()
    seasons = _season_list(first_season, _current_season())
    total   = 0
    for season in seasons:
        for stype in ("Regular Season", "Playoffs"):
            print(f"  {season} {stype}...")
            df = _fetch_season(season, stype)
            n  = _store_games(df, season)
            total += n
            print(f"    Stored {n} games.")
    print(f"Backfill complete — {total} total games.")


def fetch_recent():
    """Re-fetch the current season to pick up the latest completed games."""
    setup_db()
    season = _current_season()
    for stype in ("Regular Season", "Playoffs"):
        df = _fetch_season(season, stype)
        n  = _store_games(df, season)
        print(f"  Updated {n} {stype} games for {season}.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    backfill()
