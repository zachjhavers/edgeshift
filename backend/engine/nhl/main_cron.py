"""
NHL prediction pipeline orchestrator.

Usage:
  python main_cron.py              # daily: schedule → stats → features → predict → odds → EV
  python main_cron.py --full       # first run: backfill + train model
  python main_cron.py --retrain    # incremental update + retrain model
  python main_cron.py --pregame    # pre-game (~90 min before first puck): refresh odds + re-run EV

Cron (server example):
  # Daily predictions — 10am ET (14:00 UTC)
  0 14 * * * cd /opt/backend/engine/nhl && python3 main_cron.py >> /var/log/bots/nhl-engine.log 2>&1

  # Pre-game odds refresh — 5:30pm ET (21:30 UTC) on game nights
  30 21 * * * cd /opt/backend/engine/nhl && python3 main_cron.py --pregame >> /var/log/bots/nhl-engine.log 2>&1

  # Weekly retrain — Tuesday 1am ET (06:00 UTC)
  0 6 * * 2 cd /opt/backend/engine/nhl && python3 main_cron.py --retrain >> /var/log/bots/nhl-engine.log 2>&1

  # Result updates — 11pm ET (04:00 UTC) and 1:30am ET (06:30 UTC)
  0 4   * * * cd /opt/backend/engine/nhl && python3 -c "from update_results import update_results; update_results()" >> /var/log/bots/nhl-results.log 2>&1
  30 6  * * * cd /opt/backend/engine/nhl && python3 -c "from update_results import update_results; update_results()" >> /var/log/bots/nhl-results.log 2>&1
"""

import argparse
import os
import sys
from datetime import datetime


def _header(title: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  {title}  [{ts}]")
    print(f"{'='*60}")


def daily_update():
    from db import setup_db
    from fetch_schedule import fetch_schedule
    from fetch_stats import fetch_stats
    from feature_builder import build_features
    from predict import generate_predictions
    from update_results import update_results
    from fetch_odds import fetch_and_store_odds
    from ev_engine import run_ev

    _header("NHL Engine — Daily Update")
    errors = []

    try:
        setup_db()
        update_results()
        fetch_schedule()
        fetch_stats()
        build_features()
        preds = generate_predictions()
        if preds is None:
            print("No games today — skipping predictions.")
        fetch_and_store_odds()
        run_ev()
    except Exception as e:
        errors.append(str(e))
        print(f"ERROR — {e}")

    if errors:
        print(f"=== Daily update finished with {len(errors)} error(s) ===")
        sys.exit(1)
    else:
        print("=== Daily update done ===")


def pregame_update():
    from fetch_odds import fetch_and_store_odds
    from ev_engine import run_ev

    _header("NHL Engine — Pre-Game Odds Refresh")
    errors = []

    print("Step 1: Refreshing multi-book odds (pre-game snapshot)...")
    try:
        fetch_and_store_odds()
    except Exception as e:
        msg = f"Pre-game odds fetch failed: {e}"
        print(f"ERROR — {msg}")
        errors.append(msg)

    print("Step 2: Re-running EV engine with updated odds + line-movement filter...")
    try:
        run_ev()
    except Exception as e:
        msg = f"Pre-game EV run failed: {e}"
        print(f"ERROR — {msg}")
        errors.append(msg)

    if errors:
        print(f"=== Pre-game run finished with {len(errors)} error(s) ===")
        sys.exit(1)
    else:
        print("=== Pre-game run done ===")


def full_rebuild():
    from db import setup_db
    from fetch_schedule import fetch_schedule
    from fetch_stats import fetch_stats
    from feature_builder import build_features
    from model_builder import build_and_train_model
    from predict import generate_predictions

    _header("NHL Engine — Full Rebuild")
    setup_db()
    fetch_schedule(full=True)
    fetch_stats()
    build_features()
    build_and_train_model()
    generate_predictions()


def retrain_update():
    from db import setup_db
    from fetch_schedule import fetch_schedule
    from fetch_stats import fetch_stats
    from feature_builder import build_features
    from model_builder import build_and_train_model
    from predict import generate_predictions
    from update_results import update_results

    _header("NHL Engine — Update + Retrain")
    setup_db()
    update_results()
    fetch_schedule()
    fetch_stats()
    build_features()
    build_and_train_model()
    preds = generate_predictions()
    if preds is None:
        print("Predictions failed after retrain.")
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL prediction pipeline")
    parser.add_argument("--full",    action="store_true", help="Full rebuild from scratch")
    parser.add_argument("--retrain", action="store_true", help="Incremental update + retrain model")
    parser.add_argument("--pregame", action="store_true", help="Pre-game: refresh odds + re-run EV with line-movement filter")
    args = parser.parse_args()

    if args.full:
        full_rebuild()
    elif args.retrain:
        retrain_update()
    elif args.pregame:
        pregame_update()
    else:
        daily_update()
