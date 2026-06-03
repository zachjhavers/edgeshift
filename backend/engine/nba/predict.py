"""
Generate NBA win-probability predictions for today's games.
Loads the regular or playoff model depending on the game type detected
from the NBA game ID.
"""

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from db import get_conn
from feature_builder import get_prediction_features
from fetch_schedule import get_today_games
from utils import FEATURES

MODEL_DIR = Path(__file__).parent / "models"


def _load_model(game_type: str) -> dict | None:
    path = MODEL_DIR / f"xgb_nba_{game_type}.pkl"
    if not path.exists():
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict(bundle: dict, features: dict) -> float:
    X    = np.array([[features[f] for f in FEATURES]], dtype=float)
    raw  = bundle["xgb"].predict_proba(X)[:, 1].reshape(-1, 1)
    prob = float(np.clip(bundle["platt"].predict_proba(raw)[:, 1][0], 0.01, 0.99))
    return prob


def run_predictions(as_of_date: str | None = None) -> list[dict]:
    today  = as_of_date or datetime.now().strftime("%Y-%m-%d")
    games  = get_today_games(today)

    if not games:
        print(f"  No NBA games scheduled for {today}.")
        return []

    conn   = get_conn()
    now    = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    preds  = []

    # Load both models once
    models = {
        "regular": _load_model("regular"),
        "playoff": _load_model("playoff"),
    }

    for game in games:
        home      = game["home_team"]
        away      = game["away_team"]
        game_type = game["game_type"]
        game_id   = game["game_id"]
        matchup   = f"{away} @ {home}"

        bundle = models.get(game_type) or models.get("regular")
        if bundle is None:
            print(f"  No model available for {matchup} ({game_type}) — run model_builder first.")
            continue

        feats = get_prediction_features(home, away, today, conn)
        if feats is None:
            print(f"  Skipping {matchup} — insufficient prior game data.")
            continue

        home_prob = _predict(bundle, feats)
        away_prob = round(1.0 - home_prob, 4)
        home_prob = round(home_prob, 4)

        pred = {
            "game_date":     today,
            "game_id":       game_id,
            "home_team":     home,
            "away_team":     away,
            "game_type":     game_type,
            "home_win_prob": home_prob,
            "away_win_prob": away_prob,
            "created_at":    now,
        }
        preds.append(pred)

        conn.execute("""
            INSERT INTO predictions
                (game_date, game_id, home_team, away_team, game_type,
                 home_win_prob, away_win_prob, created_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(game_date, game_id) DO UPDATE SET
                home_win_prob = excluded.home_win_prob,
                away_win_prob = excluded.away_win_prob,
                created_at    = excluded.created_at
        """, (today, game_id, home, away, game_type,
              home_prob, away_prob, now))

    conn.commit()
    conn.close()

    if preds:
        print(f"\n{'='*70}")
        type_label = preds[0]["game_type"].upper() if preds else ""
        print(f"  NBA PREDICTIONS  —  {today}  [{type_label}]")
        print(f"{'='*70}\n")
        print(f"  {'Matchup':<28} {'Home%':>6}  {'Away%':>6}  Type")
        print(f"  {'-'*50}")
        for p in sorted(preds, key=lambda x: -max(x["home_win_prob"], x["away_win_prob"])):
            print(f"  {p['away_team']} @ {p['home_team']:<20} "
                  f"{p['home_win_prob']*100:5.1f}%  "
                  f"{p['away_win_prob']*100:5.1f}%  "
                  f"{p['game_type']}")
    else:
        print("  No predictions generated.")

    return preds


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    run_predictions()
