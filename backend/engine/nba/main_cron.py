"""
Daily cron entry point for the NBA engine.

Morning run  (10 AM ET):
  1. Resolve yesterday's TBD bets
  2. Fetch recent game results to keep DB current
  3. Rebuild features + update Elo (weekly on Mondays, or forced with --rebuild)
  4. Retrain both regular/playoff models (Mondays only)
  5. Fetch today's odds (opening snapshot)
  6. Run predictions + EV engine

Pre-game run (6 PM ET):
  1. Refresh odds (pre-game snapshot, triggers line-movement calc)
  2. Re-run EV engine with updated lines
"""

import argparse
import sys
from datetime import datetime

from ev_engine import run_ev
from feature_builder import build_all_features
from fetch_odds import fetch_and_store_odds
from fetch_stats import fetch_recent
from model_builder import build_and_train
from predict import run_predictions
from update_results import update_results


def log(msg: str):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}")


def run(pregame: bool = False, rebuild: bool = False):
    errors = []

    if pregame:
        log("=== NBA Pre-Game Odds Refresh ===")

        log("Step 1: Refreshing odds (pre-game snapshot)...")
        try:
            fetch_and_store_odds()
        except Exception as e:
            errors.append(f"Odds fetch failed: {e}")
            log(f"ERROR — {errors[-1]}")

        log("Step 2: Re-running predictions + EV engine...")
        try:
            run_predictions()
            run_ev()
        except Exception as e:
            errors.append(f"Prediction run failed: {e}")
            log(f"ERROR — {errors[-1]}")

        status = f"finished with {len(errors)} error(s): {errors}" if errors else "done"
        log(f"=== Pre-game run {status} ===")
        if errors:
            sys.exit(1)
        return

    # ── Morning run ─────────────────────────────────────────────────────────
    log("=== NBA Morning Cron ===")

    log("Step 1: Resolving yesterday's TBD bets...")
    try:
        update_results()
    except Exception as e:
        errors.append(f"Result update failed: {e}")
        log(f"ERROR — {errors[-1]}")

    log("Step 2: Fetching recent game results...")
    try:
        fetch_recent()
    except Exception as e:
        errors.append(f"Stats fetch failed: {e}")
        log(f"ERROR — {errors[-1]}")

    is_monday = datetime.now().weekday() == 0
    if is_monday or rebuild:
        log("Step 3: Rebuilding features + Elo...")
        try:
            build_all_features()
        except Exception as e:
            errors.append(f"Feature build failed: {e}")
            log(f"ERROR — {errors[-1]}")

        log("Step 4: Retraining regular + playoff models...")
        try:
            build_and_train("regular")
            build_and_train("playoff")
        except Exception as e:
            errors.append(f"Model training failed: {e}")
            log(f"ERROR — {errors[-1]}")
    else:
        log("Step 3+4: Skipping feature rebuild + retrain (runs Mondays).")

    log("Step 5: Fetching opening odds...")
    try:
        fetch_and_store_odds()
    except Exception as e:
        errors.append(f"Odds fetch failed: {e}")
        log(f"ERROR — {errors[-1]}")

    log("Step 6: Running predictions + EV engine...")
    try:
        run_predictions()
        run_ev()
    except Exception as e:
        errors.append(f"Prediction run failed: {e}")
        log(f"ERROR — {errors[-1]}")

    status = f"finished with {len(errors)} error(s): {errors}" if errors else "done"
    log(f"=== Morning run {status} ===")
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    from dotenv import load_dotenv
    from pathlib import Path
    load_dotenv(Path(__file__).parents[2] / ".env")

    parser = argparse.ArgumentParser(description="NBA daily cron")
    parser.add_argument("--pregame", action="store_true",
                        help="Pre-game mode: refresh odds + re-run EV engine")
    parser.add_argument("--rebuild", action="store_true",
                        help="Force feature rebuild + model retrain regardless of day")
    args = parser.parse_args()
    run(pregame=args.pregame, rebuild=args.rebuild)
