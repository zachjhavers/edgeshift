"""
Fetch moneyline odds from multiple bookmakers via The Odds API.

Improvements over v1:
- Requests Pinnacle + DraftKings + FanDuel + BetMGM + Caesars in one call
- Computes consensus probability (average vig-free across all available books)
- Uses Pinnacle vig-free prob as the market baseline for EV calculations
- Tracks best available odds per side across all books
- Preserves opening_pinnacle_home_prob on first daily fetch for line-movement tracking
"""

import os
from datetime import datetime

import requests
import pandas as pd
from sqlalchemy import text as _text

from db import get_engine
from utils import MLB_TEAM_MAP

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORT        = "baseball_mlb"
REGIONS      = "us"
MARKETS      = "h2h"

# All bookmakers requested in a single API call (costs one request quota unit)
BOOKMAKERS   = "pinnacle,draftkings,fanduel,betmgm,williamhill_us"
SHARP_BOOK   = "pinnacle"       # sharpest market; used for vig-free probability baseline
DISPLAY_BOOK = "draftkings"     # fallback for display odds when Pinnacle unavailable


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    """Return vig-removed implied home win probability from decimal odds."""
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    return raw_home / (raw_home + raw_away)


def _ensure_odds_table(engine) -> None:
    """Create historical_odds with all columns or add new columns to existing table."""
    with engine.begin() as conn:
        conn.execute(_text("""
            CREATE TABLE IF NOT EXISTS historical_odds (
                id                         SERIAL PRIMARY KEY,
                api_event_id               VARCHAR(50),
                game_date                  DATE        NOT NULL,
                home_team                  VARCHAR(10) NOT NULL,
                away_team                  VARCHAR(10) NOT NULL,
                home_odds                  FLOAT,
                away_odds                  FLOAT,
                pinnacle_home_odds         FLOAT,
                pinnacle_away_odds         FLOAT,
                consensus_home_prob        FLOAT,
                best_home_odds             FLOAT,
                best_away_odds             FLOAT,
                best_home_book             VARCHAR(20),
                best_away_book             VARCHAR(20),
                opening_pinnacle_home_prob FLOAT,
                bookmaker_source           VARCHAR(20),
                last_fetched_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (game_date, home_team, away_team)
            )
        """))
        # Migrate older tables that lack the new columns
        new_cols = [
            ("pinnacle_home_odds",         "FLOAT"),
            ("pinnacle_away_odds",         "FLOAT"),
            ("consensus_home_prob",        "FLOAT"),
            ("best_home_odds",             "FLOAT"),
            ("best_away_odds",             "FLOAT"),
            ("best_home_book",             "VARCHAR(20)"),
            ("best_away_book",             "VARCHAR(20)"),
            ("opening_pinnacle_home_prob", "FLOAT"),
            ("last_fetched_at",            "TIMESTAMPTZ DEFAULT NOW()"),
        ]
        for col, dtype in new_cols:
            try:
                conn.execute(_text(
                    f"ALTER TABLE historical_odds ADD COLUMN IF NOT EXISTS {col} {dtype}"
                ))
            except Exception:
                pass


def fetch_and_store_live_odds() -> None:
    """
    Fetch today's moneyline odds from multiple bookmakers and upsert into
    historical_odds.  For each game:
      - pinnacle_home/away_odds  : sharpest market (Pinnacle)
      - consensus_home_prob      : average vig-free prob across all available books
      - best_home/away_odds      : highest decimal odds for that side (any book)
      - home/away_odds            : DraftKings display odds (fallback to Pinnacle)
      - opening_pinnacle_home_prob: set on first fetch; preserved on subsequent fetches
    """
    print("Fetching multi-book market odds (Pinnacle · DraftKings · FanDuel · BetMGM · Caesars)...")

    url    = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
    params = {
        "api_key":    ODDS_API_KEY,
        "regions":    REGIONS,
        "markets":    MARKETS,
        "oddsFormat": "decimal",
        "bookmakers": BOOKMAKERS,
    }

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"  Failed to fetch odds: {e}")
        return

    events    = resp.json()
    odds_list = []

    for event in events:
        home_full = event["home_team"]
        away_full = event["away_team"]
        game_date = event["commence_time"].split("T")[0]

        book_vf_probs: list[float] = []
        pinnacle_home_odds = pinnacle_away_odds = None
        dk_home = dk_away = None
        best_home_odds = best_away_odds = 0.0
        best_home_book = best_away_book = None

        for book in event.get("bookmakers", []):
            bkey   = book["key"]
            market = next((m for m in book["markets"] if m["key"] == "h2h"), None)
            if not market:
                continue
            h = next((o["price"] for o in market["outcomes"] if o["name"] == home_full), None)
            a = next((o["price"] for o in market["outcomes"] if o["name"] == away_full), None)
            if h is None or a is None or h <= 1.0 or a <= 1.0:
                continue

            book_vf_probs.append(_vig_free_prob(h, a))

            if bkey == SHARP_BOOK:
                pinnacle_home_odds = h
                pinnacle_away_odds = a
            if bkey == DISPLAY_BOOK:
                dk_home = h
                dk_away = a

            if h > best_home_odds:
                best_home_odds, best_home_book = h, bkey
            if a > best_away_odds:
                best_away_odds, best_away_book = a, bkey

        if not book_vf_probs:
            continue

        consensus_home_prob = sum(book_vf_probs) / len(book_vf_probs)

        # Display odds: prefer DraftKings, fall back to Pinnacle, fall back to best
        display_home = dk_home or pinnacle_home_odds or (best_home_odds if best_home_odds > 0 else None)
        display_away = dk_away or pinnacle_away_odds or (best_away_odds if best_away_odds > 0 else None)
        if display_home is None or display_away is None:
            continue

        odds_list.append({
            "api_event_id":         event["id"],
            "game_date":            game_date,
            "home_team":            home_full,
            "away_team":            away_full,
            "home_odds":            display_home,
            "away_odds":            display_away,
            "pinnacle_home_odds":   pinnacle_home_odds,
            "pinnacle_away_odds":   pinnacle_away_odds,
            "consensus_home_prob":  round(consensus_home_prob, 4),
            "best_home_odds":       best_home_odds if best_home_odds > 0 else None,
            "best_away_odds":       best_away_odds if best_away_odds > 0 else None,
            "best_home_book":       best_home_book,
            "best_away_book":       best_away_book,
        })

    if not odds_list:
        print("  No active markets found for the upcoming slate.")
        return

    df = pd.DataFrame(odds_list)

    unmapped = (
        set(df["home_team"].tolist() + df["away_team"].tolist()) - set(MLB_TEAM_MAP.keys())
    )
    if unmapped:
        print(f"  Warning: unmapped team names dropped — {sorted(unmapped)}")
    df["home_team"] = df["home_team"].map(MLB_TEAM_MAP)
    df["away_team"] = df["away_team"].map(MLB_TEAM_MAP)
    df = df.dropna(subset=["home_team", "away_team", "home_odds", "away_odds"])
    if df.empty:
        print("  No valid rows after team mapping.")
        return

    engine = get_engine()
    _ensure_odds_table(engine)

    with engine.begin() as conn:
        for _, row in df.iterrows():
            # Compute Pinnacle vig-free prob for this row (used as opening baseline)
            pin_home_prob: float | None = None
            if pd.notna(row.get("pinnacle_home_odds")) and pd.notna(row.get("pinnacle_away_odds")):
                pin_home_prob = round(
                    _vig_free_prob(float(row["pinnacle_home_odds"]), float(row["pinnacle_away_odds"])), 4
                )

            conn.execute(_text("""
                INSERT INTO historical_odds
                    (api_event_id, game_date, home_team, away_team,
                     home_odds, away_odds,
                     pinnacle_home_odds, pinnacle_away_odds,
                     consensus_home_prob,
                     best_home_odds, best_away_odds,
                     best_home_book, best_away_book,
                     opening_pinnacle_home_prob,
                     bookmaker_source, last_fetched_at)
                VALUES
                    (:api_event_id, :game_date, :home_team, :away_team,
                     :home_odds, :away_odds,
                     :pinnacle_home_odds, :pinnacle_away_odds,
                     :consensus_home_prob,
                     :best_home_odds, :best_away_odds,
                     :best_home_book, :best_away_book,
                     :opening_pinnacle_home_prob,
                     'multi', NOW())
                ON CONFLICT (game_date, home_team, away_team) DO UPDATE SET
                    api_event_id            = EXCLUDED.api_event_id,
                    home_odds               = EXCLUDED.home_odds,
                    away_odds               = EXCLUDED.away_odds,
                    pinnacle_home_odds      = EXCLUDED.pinnacle_home_odds,
                    pinnacle_away_odds      = EXCLUDED.pinnacle_away_odds,
                    consensus_home_prob     = EXCLUDED.consensus_home_prob,
                    best_home_odds          = EXCLUDED.best_home_odds,
                    best_away_odds          = EXCLUDED.best_away_odds,
                    best_home_book          = EXCLUDED.best_home_book,
                    best_away_book          = EXCLUDED.best_away_book,
                    opening_pinnacle_home_prob = COALESCE(
                        historical_odds.opening_pinnacle_home_prob,
                        EXCLUDED.opening_pinnacle_home_prob
                    ),
                    bookmaker_source        = 'multi',
                    last_fetched_at         = NOW()
            """), {
                "api_event_id":             row.get("api_event_id"),
                "game_date":                row["game_date"],
                "home_team":                row["home_team"],
                "away_team":                row["away_team"],
                "home_odds":                float(row["home_odds"]),
                "away_odds":                float(row["away_odds"]),
                "pinnacle_home_odds":       float(row["pinnacle_home_odds"]) if pd.notna(row.get("pinnacle_home_odds")) else None,
                "pinnacle_away_odds":       float(row["pinnacle_away_odds"]) if pd.notna(row.get("pinnacle_away_odds")) else None,
                "consensus_home_prob":      float(row["consensus_home_prob"]),
                "best_home_odds":           float(row["best_home_odds"]) if pd.notna(row.get("best_home_odds")) else None,
                "best_away_odds":           float(row["best_away_odds"]) if pd.notna(row.get("best_away_odds")) else None,
                "best_home_book":           row.get("best_home_book"),
                "best_away_book":           row.get("best_away_book"),
                "opening_pinnacle_home_prob": pin_home_prob,
            })

    print(f"  Stored odds for {len(df)} game(s) across {len(events)} event(s).")


if __name__ == "__main__":
    fetch_and_store_live_odds()
