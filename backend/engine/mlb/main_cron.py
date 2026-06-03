"""
Daily cron entry point for the MLB engine. Run each morning before first pitch.

Cron schedule (server example):
  # Morning run: fetch opening odds + generate predictions
  30 9 * * * cd /opt/backend/engine/mlb && python3 main_cron.py >> /var/log/bots/mlb-engine.log 2>&1

  # Pre-game snapshot: refresh odds ~90 min before first pitch for line-movement data
  30 17 * * * cd /opt/backend/engine/mlb && python3 main_cron.py --pregame >> /var/log/bots/mlb-engine.log 2>&1

  # Results: runs 4× each evening as games finish, plus 1:30am for late West Coast games
  0 19,21,23 * * * cd /opt/backend/engine/mlb && python3 -c "from update_results import update_results; update_results()" >> /var/log/bots/mlb-results.log 2>&1
  30 1 * * *       cd /opt/backend/engine/mlb && python3 -c "from update_results import update_results; update_results()" >> /var/log/bots/mlb-results.log 2>&1

Steps (morning run):
  1. Update results for prior TBD predictions + resolve CLV
  2. Fetch yesterday's Statcast data
  3. Retrain moneyline model (Mondays only)
  4. Fetch multi-book odds (morning snapshot — sets opening_pinnacle_home_prob)
  5. Run prediction engine

Steps (--pregame run, ~90 min before first pitch):
  4. Refresh multi-book odds (pre-game snapshot — updates lines for line-movement calc)
  5. Re-run prediction engine with updated odds + line-movement filter
"""

import argparse
import os
import sys
from datetime import datetime, timedelta

from db import get_engine
from ev_engine import run_predictions
from fetch_odds import fetch_and_store_live_odds, fetch_and_store_totals_odds
from fetch_onfield import fetch_and_store_games, get_fetched_dates
from model_builder import build_and_train_model
from totals_ev_engine import run_totals_predictions
from update_results import update_results


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run(pregame: bool = False):
    if pregame:
        log("=== MLB Pre-Game Odds Refresh ===")
        errors = []

        log("Step 1: Refreshing multi-book odds (pre-game snapshot)...")
        try:
            fetch_and_store_live_odds()
        except Exception as e:
            msg = f"Pre-game odds fetch failed: {e}"
            log(f"ERROR — {msg}")
            errors.append(msg)

        log("Step 2: Re-running prediction engine with updated odds + line-movement filter...")
        try:
            run_predictions()
        except Exception as e:
            msg = f"Pre-game prediction run failed: {e}"
            log(f"ERROR — {msg}")
            errors.append(msg)

        log("Step 3: Refreshing totals odds + running totals EV engine...")
        try:
            fetch_and_store_totals_odds()
            run_totals_predictions()
        except Exception as e:
            msg = f"Totals pre-game run failed: {e}"
            log(f"ERROR — {msg}")
            errors.append(msg)

        if errors:
            log(f"=== Pre-game run finished with {len(errors)} error(s): {errors} ===")
            sys.exit(1)
        else:
            log("=== Pre-game run done ===")
        return

    # ── Morning run ─────────────────────────────────────────────────────────
    log("=== MLB Prediction Cron Starting (morning) ===")
    errors = []

    log("Step 1: Updating results for pending predictions + CLV resolution...")
    try:
        update_results()
    except Exception as e:
        msg = f"Result update failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"Step 2: Fetching Statcast data for {yesterday}...")
    try:
        engine  = get_engine()
        fetched = get_fetched_dates(engine)
        fetch_and_store_games(yesterday, engine, fetched)
    except Exception as e:
        msg = f"Statcast fetch failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    if datetime.now().weekday() == 0:
        log("Step 3: Monday — retraining moneyline model (Brier-optimised Platt)...")
        try:
            build_and_train_model()
        except Exception as e:
            msg = f"Model training failed: {e}"
            log(f"ERROR — {msg}")
            errors.append(msg)
    else:
        log("Step 3: Skipping model retrain (runs on Mondays).")

    log("Step 4: Fetching multi-book odds (morning snapshot — sets opening Pinnacle line)...")
    try:
        fetch_and_store_live_odds()
    except Exception as e:
        msg = f"Odds fetch failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    log("Step 4b: Fetching totals odds (morning snapshot)...")
    try:
        fetch_and_store_totals_odds()
    except Exception as e:
        msg = f"Totals odds fetch failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    log("Step 5: Running moneyline prediction engine...")
    try:
        run_predictions()
    except Exception as e:
        msg = f"Prediction engine failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    log("Step 6: Running totals EV engine...")
    try:
        run_totals_predictions()
    except Exception as e:
        msg = f"Totals EV engine failed: {e}"
        log(f"ERROR — {msg}")
        errors.append(msg)

    if errors:
        log(f"=== Finished with {len(errors)} error(s): {errors} ===")
        sys.exit(1)
    else:
        log("=== Done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLB daily cron")
    parser.add_argument(
        "--pregame", action="store_true",
        help="Pre-game mode: refresh odds + re-run predictions with line-movement filter"
    )
    args = parser.parse_args()
    run(pregame=args.pregame)
