"""
Download and store historical international football results from the
martj42/international_results GitHub dataset (CSV updated to present day).

Run once (or monthly) to backfill / refresh the matches table.
"""

import io
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests

from db import get_conn, setup_db
from utils import importance_weight

CSV_URL = (
    "https://raw.githubusercontent.com/martj42/international_results"
    "/master/results.csv"
)

# Only use matches from this cutoff — older data is too stale for team strengths
HISTORY_CUTOFF = "2018-01-01"


def fetch_and_store(cutoff: str = HISTORY_CUTOFF) -> int:
    print(f"Downloading international results from GitHub...")
    try:
        resp = requests.get(CSV_URL, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to download: {e}")
        return 0

    df = pd.read_csv(io.StringIO(resp.text))
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df[df["date"] >= cutoff].copy()

    # Only completed matches
    df = df.dropna(subset=["home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Derive result
    def _result(row) -> str:
        if row["home_score"] > row["away_score"]:
            return "HOME_WIN"
        elif row["home_score"] < row["away_score"]:
            return "AWAY_WIN"
        return "DRAW"

    df["result"]  = df.apply(_result, axis=1)
    df["neutral"] = df["neutral"].astype(int)
    df["tournament"] = df["tournament"].fillna("Friendly")

    conn = get_conn()
    stored = 0
    for _, row in df.iterrows():
        try:
            conn.execute("""
                INSERT INTO matches
                    (match_date, home_team, away_team, home_score, away_score,
                     tournament, neutral, result)
                VALUES (?,?,?,?,?,?,?,?)
                ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
                    home_score = excluded.home_score,
                    away_score = excluded.away_score,
                    result     = excluded.result
            """, (
                row["date"], row["home_team"], row["away_team"],
                row["home_score"], row["away_score"],
                row["tournament"], row["neutral"], row["result"],
            ))
            stored += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    print(f"  Stored {stored} matches (since {cutoff}).")
    return stored


def store_wc_result(match_date: str, home_team: str, away_team: str,
                    home_score: int, away_score: int,
                    tournament: str = "FIFA World Cup") -> None:
    """Upsert a single completed World Cup match result."""
    if home_score > away_score:
        result = "HOME_WIN"
    elif home_score < away_score:
        result = "AWAY_WIN"
    else:
        result = "DRAW"

    conn = get_conn()
    conn.execute("""
        INSERT INTO matches
            (match_date, home_team, away_team, home_score, away_score,
             tournament, neutral, result)
        VALUES (?,?,?,?,?,?,1,?)
        ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
            home_score = excluded.home_score,
            away_score = excluded.away_score,
            result     = excluded.result
    """, (match_date, home_team, away_team, home_score, away_score,
          tournament, result))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    setup_db()
    fetch_and_store()
