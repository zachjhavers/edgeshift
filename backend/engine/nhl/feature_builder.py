"""
Compute the 35-feature row for every game in the DB that doesn't yet have features.

Features are built from completed prior games only (no lookahead).
Run after fetch_schedule + fetch_stats have populated raw tables.
"""

import pandas as pd

from db import get_conn
from feature_helpers import (
    build_elo_lookup,
    get_goalie_rolling_stats,
    get_rest_days,
    get_starting_goalie_id,
    get_team_rolling_stats,
)
from utils import FEATURES


def _load_tables():
    conn = get_conn()
    df_games   = pd.read_sql("SELECT * FROM games   ORDER BY game_date", conn)
    df_stats   = pd.read_sql("SELECT * FROM team_game_stats", conn)
    df_goalies = pd.read_sql("SELECT * FROM goalie_game_stats", conn)
    conn.close()
    return df_games, df_stats, df_goalies


def _games_needing_features(df_games: pd.DataFrame) -> pd.DataFrame:
    conn = get_conn()
    done = set(
        r[0] for r in conn.execute("SELECT game_id FROM features").fetchall()
    )
    conn.close()
    completed = df_games[df_games["result"].isin(["HOME_WIN", "AWAY_WIN"])]
    return completed[~completed["game_id"].astype(str).isin(done)]


def build_features():
    print("Loading tables...")
    df_games, df_stats, df_goalies = _load_tables()

    # Elo lookup for all completed games
    print("Computing Elo ratings...")
    elo_lookup = build_elo_lookup(df_games)

    pending = _games_needing_features(df_games)
    print(f"Building features for {len(pending)} games...")

    rows = []
    skipped = 0

    for i, (_, game) in enumerate(pending.sort_values("game_date").iterrows(), 1):
        game_id   = str(game["game_id"])
        game_date = str(game["game_date"])
        home      = str(game["home_team"])
        away      = str(game["away_team"])
        result    = str(game["result"])
        home_win  = 1 if result == "HOME_WIN" else 0
        game_type = int(game.get("game_type", 2))
        month     = int(game_date[5:7])

        # Team rolling stats
        h_stats = get_team_rolling_stats(home, game_date, df_games, df_stats)
        a_stats = get_team_rolling_stats(away, game_date, df_games, df_stats)
        if h_stats is None or a_stats is None:
            skipped += 1
            continue

        # Goalie rolling stats
        h_goalie_id = get_starting_goalie_id(home, game_id, df_goalies)
        a_goalie_id = get_starting_goalie_id(away, game_id, df_goalies)

        h_g = get_goalie_rolling_stats(h_goalie_id, game_date, df_games, df_goalies) if h_goalie_id else None
        a_g = get_goalie_rolling_stats(a_goalie_id, game_date, df_games, df_goalies) if a_goalie_id else None

        # Fall back to team SV% if goalie history is unavailable
        h_goalie_sv  = h_g["sv_pct"] if h_g else h_stats["sv_pct"]
        h_goalie_gsaa = h_g["gsaa"]  if h_g else 0.0
        a_goalie_sv  = a_g["sv_pct"] if a_g else a_stats["sv_pct"]
        a_goalie_gsaa = a_g["gsaa"]  if a_g else 0.0

        # Rest days
        h_rest = get_rest_days(home, game_date, df_games)
        a_rest = get_rest_days(away, game_date, df_games)
        h_b2b  = 1 if h_rest == 1 else 0
        a_b2b  = 1 if a_rest == 1 else 0

        # Elo
        home_elo_prob = elo_lookup.get(game_id, 0.5)
        elo_diff      = home_elo_prob - (1 - home_elo_prob)

        row = (
            game_id,
            # Team rolling
            h_stats["gf"],  a_stats["gf"],
            h_stats["ga"],  a_stats["ga"],
            h_stats["sf"],  a_stats["sf"],
            h_stats["sa"],  a_stats["sa"],
            h_stats["shot_pct"], a_stats["shot_pct"],
            h_stats["sv_pct"],   a_stats["sv_pct"],
            h_stats["pp_pct"],   a_stats["pp_pct"],
            h_stats["pk_pct"],   a_stats["pk_pct"],
            h_stats["win_pct"],  a_stats["win_pct"],
            # Differentials
            h_stats["gf"] - a_stats["gf"],
            h_stats["sf"] - a_stats["sf"],
            h_stats["shot_pct"] - a_stats["shot_pct"],
            h_stats["sv_pct"]   - a_stats["sv_pct"],
            h_stats["pp_pct"]   - a_stats["pp_pct"],
            h_stats["pk_pct"]   - a_stats["pk_pct"],
            h_stats["win_pct"]  - a_stats["win_pct"],
            # Goalie
            h_goalie_sv,   a_goalie_sv,
            h_goalie_gsaa, a_goalie_gsaa,
            h_goalie_sv - a_goalie_sv,
            h_goalie_gsaa - a_goalie_gsaa,
            # Rest
            h_rest, a_rest, h_b2b, a_b2b, h_rest - a_rest,
            # Elo
            round(home_elo_prob, 4), round(elo_diff, 4),
            # Calendar
            1 if game_type == 3 else 0,
            month,
            # PDO and shot share
            h_stats["pdo"],        a_stats["pdo"],
            round(h_stats["pdo"] - a_stats["pdo"], 4),
            h_stats["shot_share"], a_stats["shot_share"],
            round(h_stats["shot_share"] - a_stats["shot_share"], 4),
            # Target
            home_win,
        )
        rows.append(row)

        if i % 500 == 0:
            print(f"  {i}/{len(pending)} processed...")

    if not rows:
        print(f"No new features to write (skipped {skipped} games with insufficient history).")
        return

    placeholders = ",".join(["?"] * (len(FEATURES) + 2))  # game_id + features + home_win
    cols = "game_id," + ",".join(FEATURES) + ",home_win"

    conn = get_conn()
    conn.executemany(
        f"INSERT OR REPLACE INTO features ({cols}) VALUES ({placeholders})",
        rows,
    )
    conn.commit()
    conn.close()
    print(f"Done: {len(rows)} feature rows written, {skipped} skipped (insufficient history).")


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    build_features()
