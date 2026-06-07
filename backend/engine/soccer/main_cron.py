"""
Soccer (FIFA World Cup) prediction pipeline.

Usage:
  python main_cron.py              # daily: schedule → odds → predict → EV
  python main_cron.py --full       # first run: fetch history + train model
  python main_cron.py --retrain    # retrain model on latest data
  python main_cron.py --pregame    # pre-game: refresh odds + re-run EV
"""

import argparse
import sys
from datetime import datetime


def _header(title: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"\n{'='*60}")
    print(f"  {title}  [{ts}]")
    print(f"{'='*60}")


def daily_update():
    from db import setup_db
    from fetch_schedule import fetch_and_store_schedule
    from fetch_odds import fetch_and_store_odds
    from update_results import update_results
    from predict import generate_predictions
    from ev_engine import run_ev

    _header("Soccer Engine — Daily Update")

    setup_db()
    print("Step 1: Resolving yesterday's results...")
    update_results()

    print("Step 2: Fetching upcoming schedule...")
    fetch_and_store_schedule(days_ahead=3)

    print("Step 3: Fetching odds...")
    fetch_and_store_odds()

    print("Step 4: Generating predictions...")
    preds = generate_predictions()
    if not preds:
        print("  No matches today.")

    print("Step 5: Running EV engine...")
    run_ev()

    print("=== Soccer daily update done ===")


def pregame_update():
    from fetch_odds import fetch_and_store_odds
    from ev_engine import run_ev

    _header("Soccer Engine — Pre-Game Refresh")

    print("Step 1: Refreshing odds...")
    fetch_and_store_odds()

    print("Step 2: Re-running EV engine...")
    run_ev()

    print("=== Pre-game run done ===")


def full_rebuild():
    from db import setup_db
    from fetch_history import fetch_and_store
    from model_builder import build_and_train_model

    _header("Soccer Engine — Full Rebuild")

    setup_db()
    print("Step 1: Fetching historical match data...")
    n = fetch_and_store()
    print(f"  {n} matches loaded.")

    print("Step 2: Training Dixon-Coles model...")
    bundle = build_and_train_model()
    print(f"  Model trained on {bundle['n_matches']} matches, {len(bundle['teams'])} teams.")

    print("=== Full rebuild done ===")


def retrain():
    from fetch_history import fetch_and_store
    from model_builder import build_and_train_model

    _header("Soccer Engine — Retrain")

    print("Step 1: Refreshing match data...")
    fetch_and_store()

    print("Step 2: Retraining model...")
    build_and_train_model()

    print("=== Retrain done ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--full",     action="store_true")
    parser.add_argument("--retrain",  action="store_true")
    parser.add_argument("--pregame",  action="store_true")
    args = parser.parse_args()

    if args.full:
        full_rebuild()
    elif args.retrain:
        retrain()
    elif args.pregame:
        pregame_update()
    else:
        daily_update()
