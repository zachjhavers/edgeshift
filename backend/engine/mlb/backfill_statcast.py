"""
Historical Statcast backfill — pulls missing seasons into PostgreSQL.

Fetches data month-by-month via pybaseball (more efficient than day-by-day;
pybaseball handles the chunking internally and caches results).

Already-fetched dates are skipped automatically via get_fetched_dates().

Usage:
  python3 backfill_statcast.py           # all missing seasons (2019-2024)
  python3 backfill_statcast.py --dry-run # print plan, no fetching

Run time: ~3-6 hours depending on pybaseball rate limiting.
Run in background: nohup python3 backfill_statcast.py >> /var/log/bots/mlb-backfill.log 2>&1 &
"""

import sys
import time
from datetime import datetime

import pandas as pd
import pybaseball as pyb
from sqlalchemy import text

from db import get_engine
from fetch_onfield import get_fetched_dates

# Regular season date ranges per season.
# 2020 was COVID-shortened (60 games, July–September).
SEASONS = [
    ("2019", "2019-03-28", "2019-09-30"),
    ("2020", "2020-07-23", "2020-09-27"),
    ("2021", "2021-04-01", "2021-10-03"),
    ("2022", "2022-04-07", "2022-10-05"),
    ("2023", "2023-03-30", "2023-10-01"),
    ("2024", "2024-03-20", "2024-09-29"),
]

CHUNK_WEEKS = 4   # fetch 4 weeks at a time — safe for pybaseball without timeouts


def _already_have_season(season: str, fetched: set) -> bool:
    """Check if we already have substantial data for this season."""
    season_dates = {d for d in fetched if d.startswith(season)}
    return len(season_dates) > 100  # >100 dates = season is substantially loaded


def fetch_chunk(start: str, end: str, engine, dry_run: bool) -> int:
    """Fetch one chunk of Statcast data and append to DB. Returns rows inserted."""
    print(f"    Fetching {start} → {end} ...", flush=True)
    if dry_run:
        return 0
    try:
        df = pyb.statcast(start_dt=start, end_dt=end)
        if df is None or df.empty:
            print(f"    No data returned (off-days / Statcast gap).")
            return 0
        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=["game_date", "home_team", "away_team"])
        df = df[df["game_type"] == "R"]  # regular season only
        if df.empty:
            return 0
        df.to_sql("statcast_raw", engine, if_exists="append", index=False)
        print(f"    {len(df):,} rows inserted.", flush=True)
        return len(df)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)
        return 0


def run_backfill(dry_run: bool = False):
    engine  = get_engine()
    fetched = get_fetched_dates(engine)
    print(f"DB already has {len(fetched)} distinct date(s) in statcast_raw.\n")

    total_rows = 0
    for season, start, end in SEASONS:
        if _already_have_season(season, fetched):
            print(f"  Season {season}: already loaded ({sum(1 for d in fetched if d.startswith(season))} dates) — skipping.")
            continue

        print(f"\n  Season {season}: {start} → {end}")
        chunks = list(pd.date_range(start=start, end=end, freq=f"{CHUNK_WEEKS}W"))
        if not chunks or chunks[-1] < pd.Timestamp(end):
            chunks.append(pd.Timestamp(end))

        prev = pd.Timestamp(start)
        for chunk_end in chunks:
            s = prev.strftime("%Y-%m-%d")
            e = chunk_end.strftime("%Y-%m-%d")

            # Skip entirely if we have most of these dates already
            chunk_dates = pd.date_range(s, e)
            already_have = sum(1 for d in chunk_dates if d.strftime("%Y-%m-%d") in fetched)
            if already_have >= len(chunk_dates) * 0.8:
                print(f"    {s} → {e}: {already_have}/{len(chunk_dates)} dates already in DB — skipping.")
                prev = chunk_end + pd.Timedelta(days=1)
                continue

            rows = fetch_chunk(s, e, engine, dry_run)
            total_rows += rows
            prev = chunk_end + pd.Timedelta(days=1)
            time.sleep(1)  # polite pause between chunks

        print(f"  Season {season} done.", flush=True)

    print(f"\nBackfill complete. {total_rows:,} total rows inserted.")


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv
    if dry_run:
        print("[DRY RUN] — no data will be fetched.\n")
    run_backfill(dry_run=dry_run)
