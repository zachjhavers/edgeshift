import pybaseball as pyb
import pandas as pd
from datetime import datetime
from sqlalchemy import text
from db import get_engine

def get_fetched_dates(engine) -> set:
    """Returns the set of game_dates already in the DB so we can skip them."""
    with engine.connect() as conn:
        result = conn.execute(text("SELECT DISTINCT CAST(game_date AS DATE) FROM statcast_raw"))
        return {str(row[0]) for row in result}

def fetch_and_store_games(target_date: str, engine, fetched_dates: set):
    if target_date in fetched_dates:
        print(f"  {target_date} already in DB — skipping.")
        return

    print(f"  Fetching {target_date}...")
    try:
        df = pyb.statcast(start_dt=target_date, end_dt=target_date)

        if df is None or df.empty:
            print(f"  No data for {target_date} (off-day or Statcast gap).")
            return

        df.columns = [c.lower() for c in df.columns]
        df = df.dropna(subset=['game_date', 'home_team', 'away_team'])

        df.to_sql('statcast_raw', engine, if_exists='append', index=False)
        print(f"  Loaded {len(df):,} rows.")

    except Exception as e:
        print(f"  Failed for {target_date}: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default="2026-03-20",
                        help="Start date YYYY-MM-DD (default: 2026 Opening Day)")
    parser.add_argument('--end',   default=datetime.now().strftime('%Y-%m-%d'),
                        help="End date YYYY-MM-DD (default: today)")
    args = parser.parse_args()

    engine = get_engine()
    fetched_dates = get_fetched_dates(engine)
    print(f"DB already has {len(fetched_dates)} date(s). Fetching {args.start} → {args.end}...\n")

    for date in pd.date_range(start=args.start, end=args.end):
        fetch_and_store_games(date.strftime('%Y-%m-%d'), engine, fetched_dates)
