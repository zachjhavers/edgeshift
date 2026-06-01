"""
Daily inference: for each of today's scheduled NHL games, compute the
home win probability and write it to the predictions table.

Requires:
  - Trained model bundle at models/xgb_nhl_v1.pkl
  - Games for today already in the games table (via fetch_schedule)
  - Stats and features for prior games (for rolling window computation)
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from db import get_conn
from feature_helpers import (
    compute_current_elo,
    get_goalie_rolling_stats,
    get_rest_days,
    get_starting_goalie_id,
    get_team_rolling_stats,
)
from utils import FEATURES, ELO_HOME_ADV, ELO_INIT, LEAGUE_AVG_SV_PCT

MODEL_PATH = Path(__file__).parent / "models" / "xgb_nhl_v1.pkl"

HIGH_CONF = 0.62
MED_CONF  = 0.57


def _load_bundle() -> dict | None:
    if not MODEL_PATH.exists():
        print("No trained model found. Run model_builder.py first.")
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _todays_games(today: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT game_id, home_team, away_team, game_type FROM games WHERE game_date = ? ORDER BY game_id",
        (today,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def generate_predictions(as_of_date: str | None = None) -> pd.DataFrame | None:
    bundle = _load_bundle()
    if bundle is None:
        return None

    today = as_of_date or datetime.now().strftime("%Y-%m-%d")
    games = _todays_games(today)

    if not games:
        print(f"No games found for {today}.")
        return None

    print(f"Running predictions for {len(games)} game(s) on {today}...")

    # Load all historical data needed for rolling stats
    conn = get_conn()
    df_all_games = pd.read_sql(
        "SELECT game_id, game_date, season, home_team, away_team, result, game_type FROM games ORDER BY game_date",
        conn,
    )
    df_stats   = pd.read_sql("SELECT * FROM team_game_stats", conn)
    df_goalies = pd.read_sql("SELECT * FROM goalie_game_stats", conn)
    conn.close()

    # Current Elo ratings
    elo_ratings = compute_current_elo(df_all_games[df_all_games["game_date"] < today])

    def _elo_prob(home: str, away: str) -> float:
        h = elo_ratings.get(home, ELO_INIT) + ELO_HOME_ADV
        a = elo_ratings.get(away, ELO_INIT)
        return 1.0 / (1.0 + 10 ** ((a - h) / 400))

    feature_rows = []
    game_meta    = []

    for g in games:
        game_id   = str(g["game_id"])
        home      = str(g["home_team"])
        away      = str(g["away_team"])
        game_type = int(g["game_type"])

        h_stats = get_team_rolling_stats(home, today, df_all_games, df_stats)
        a_stats = get_team_rolling_stats(away, today, df_all_games, df_stats)

        if h_stats is None or a_stats is None:
            print(f"  {game_id} ({home} vs {away}): skipped — insufficient history")
            continue

        # Goalie stats — use most recently recorded starter for each team
        recent_h = df_all_games[
            ((df_all_games["home_team"] == home) | (df_all_games["away_team"] == home)) &
            (df_all_games["game_date"] < today) &
            df_all_games["result"].isin(["HOME_WIN", "AWAY_WIN"])
        ].sort_values("game_date")
        recent_a = df_all_games[
            ((df_all_games["home_team"] == away) | (df_all_games["away_team"] == away)) &
            (df_all_games["game_date"] < today) &
            df_all_games["result"].isin(["HOME_WIN", "AWAY_WIN"])
        ].sort_values("game_date")

        h_goalie_id = None
        if not recent_h.empty:
            h_goalie_id = get_starting_goalie_id(home, str(recent_h.iloc[-1]["game_id"]), df_goalies)
        a_goalie_id = None
        if not recent_a.empty:
            a_goalie_id = get_starting_goalie_id(away, str(recent_a.iloc[-1]["game_id"]), df_goalies)

        h_g = get_goalie_rolling_stats(h_goalie_id, today, df_all_games, df_goalies) if h_goalie_id else None
        a_g = get_goalie_rolling_stats(a_goalie_id, today, df_all_games, df_goalies) if a_goalie_id else None

        h_goalie_sv   = h_g["sv_pct"] if h_g else h_stats["sv_pct"]
        h_goalie_gsaa = h_g["gsaa"]   if h_g else 0.0
        a_goalie_sv   = a_g["sv_pct"] if a_g else a_stats["sv_pct"]
        a_goalie_gsaa = a_g["gsaa"]   if a_g else 0.0

        h_rest = get_rest_days(home, today, df_all_games)
        a_rest = get_rest_days(away, today, df_all_games)
        h_b2b  = 1 if h_rest == 1 else 0
        a_b2b  = 1 if a_rest == 1 else 0

        home_elo_prob = _elo_prob(home, away)
        elo_diff      = home_elo_prob - (1 - home_elo_prob)
        month         = int(today[5:7])

        feat = [
            h_stats["gf"],  a_stats["gf"],
            h_stats["ga"],  a_stats["ga"],
            h_stats["sf"],  a_stats["sf"],
            h_stats["sa"],  a_stats["sa"],
            h_stats["shot_pct"], a_stats["shot_pct"],
            h_stats["sv_pct"],   a_stats["sv_pct"],
            h_stats["pp_pct"],   a_stats["pp_pct"],
            h_stats["pk_pct"],   a_stats["pk_pct"],
            h_stats["win_pct"],  a_stats["win_pct"],
            h_stats["gf"] - a_stats["gf"],
            h_stats["sf"] - a_stats["sf"],
            h_stats["shot_pct"] - a_stats["shot_pct"],
            h_stats["sv_pct"]   - a_stats["sv_pct"],
            h_stats["pp_pct"]   - a_stats["pp_pct"],
            h_stats["pk_pct"]   - a_stats["pk_pct"],
            h_stats["win_pct"]  - a_stats["win_pct"],
            h_goalie_sv,   a_goalie_sv,
            h_goalie_gsaa, a_goalie_gsaa,
            h_goalie_sv - a_goalie_sv,
            h_goalie_gsaa - a_goalie_gsaa,
            h_rest, a_rest, h_b2b, a_b2b, h_rest - a_rest,
            round(home_elo_prob, 4), round(elo_diff, 4),
            1 if game_type == 3 else 0,
            month,
            # PDO and shot share
            h_stats["pdo"],        a_stats["pdo"],
            round(h_stats["pdo"] - a_stats["pdo"], 4),
            h_stats["shot_share"], a_stats["shot_share"],
            round(h_stats["shot_share"] - a_stats["shot_share"], 4),
        ]

        feature_rows.append(feat)
        game_meta.append({"game_id": game_id, "home": home, "away": away})

    if not feature_rows:
        print("No games could be predicted (all skipped).")
        return None

    X   = np.array(feature_rows, dtype=float)
    xgb   = bundle["xgb"]
    platt = bundle["platt"]

    raw  = xgb.predict_proba(X)[:, 1].reshape(-1, 1)
    prob = np.clip(platt.predict_proba(raw)[:, 1], 0.0, 1.0)

    results = []
    conn = get_conn()
    for i, meta in enumerate(game_meta):
        home_prob = float(prob[i])
        away_prob = round(1 - home_prob, 4)
        home_prob = round(home_prob, 4)

        conn.execute(
            "INSERT OR REPLACE INTO predictions (game_id, prediction_date, home_win_prob, away_win_prob) "
            "VALUES (?,?,?,?)",
            (meta["game_id"], today, home_prob, away_prob),
        )
        results.append({**meta, "home_win_prob": home_prob, "away_win_prob": away_prob})

    conn.commit()
    conn.close()

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"NHL Predictions  |  {today}")
    print(f"Model: {bundle.get('train_end','?')}  |  Val acc: {bundle.get('val_accuracy',0):.1%}")
    print(f"{'='*60}")
    for r in sorted(results, key=lambda x: -x["home_win_prob"]):
        home_pct = r["home_win_prob"] * 100
        away_pct = r["away_win_prob"] * 100
        conf = "★" if home_pct >= HIGH_CONF * 100 or away_pct >= HIGH_CONF * 100 else " "
        fav  = r["home"] if home_pct >= 50 else r["away"]
        pct  = max(home_pct, away_pct)
        print(f"  {conf} {r['home']:<4} vs {r['away']:<4}  →  {fav} {pct:.1f}%  "
              f"(home {home_pct:.1f}% / away {away_pct:.1f}%)")

    return pd.DataFrame(results)


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    generate_predictions()
