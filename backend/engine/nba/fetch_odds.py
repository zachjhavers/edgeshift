"""
Fetch NBA moneyline odds from The Odds API — Pinnacle + 4 US books in one call.
Pinnacle is stored as the market baseline; best odds across all books are tracked
for display (Pinnacle typically wins due to lowest vig).
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import requests

from utils import TEAM_NAME_MAP

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORT        = "basketball_nba"
REGIONS      = "us"
MARKETS      = "h2h"
BOOKMAKERS   = "pinnacle,draftkings,fanduel,betmgm,williamhill_us"
SHARP_BOOK   = "pinnacle"

DB_PATH = Path(os.getenv("NBA_DB_PATH", str(Path(__file__).parent / "nba_predictor.db")))


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    raw_h = 1.0 / home_odds
    raw_a = 1.0 / away_odds
    return raw_h / (raw_h + raw_a)


def fetch_and_store_odds():
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set — skipping NBA odds fetch.")
        return

    print("Fetching NBA odds (Pinnacle · DraftKings · FanDuel · BetMGM · Caesars)...")
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
        print(f"  Failed to fetch NBA odds: {e}")
        return

    events     = resp.json()
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    stored     = 0

    conn = _conn()

    for event in events:
        home_full = event["home_team"]
        away_full = event["away_team"]
        home_abbr = TEAM_NAME_MAP.get(home_full)
        away_abbr = TEAM_NAME_MAP.get(away_full)

        if not home_abbr or not away_abbr:
            continue

        game_date  = event["commence_time"].split("T")[0]
        book_odds: dict[str, tuple[float, float]] = {}

        for book in event.get("bookmakers", []):
            market = next((m for m in book["markets"] if m["key"] == "h2h"), None)
            if not market:
                continue
            h = next((o["price"] for o in market["outcomes"] if o["name"] == home_full), None)
            a = next((o["price"] for o in market["outcomes"] if o["name"] == away_full), None)
            if h and a and float(h) > 1.0 and float(a) > 1.0:
                book_odds[book["key"]] = (float(h), float(a))

        if not book_odds:
            continue

        pin        = book_odds.get(SHARP_BOOK)
        pin_h      = pin[0] if pin else None
        pin_a      = pin[1] if pin else None
        vf_probs   = [_vig_free_prob(h, a) for h, a in book_odds.values()]
        consensus  = round(sum(vf_probs) / len(vf_probs), 4)
        opening_pp = round(_vig_free_prob(pin_h, pin_a), 4) if pin_h and pin_a else None

        best_h_odds = max(h for h, _ in book_odds.values())
        best_h_book = next(k for k, (h, _) in book_odds.items() if h == best_h_odds)
        best_a_odds = max(a for _, a in book_odds.values())
        best_a_book = next(k for k, (_, a) in book_odds.items() if a == best_a_odds)

        dk      = book_odds.get("draftkings")
        disp_h  = dk[0] if dk else (pin_h or best_h_odds)
        disp_a  = dk[1] if dk else (pin_a or best_a_odds)

        try:
            conn.execute("""
                INSERT INTO historical_odds (
                    game_date, home_team, away_team,
                    home_odds, away_odds,
                    pinnacle_home_odds, pinnacle_away_odds,
                    best_home_odds, best_away_odds, best_home_book, best_away_book,
                    consensus_home_prob, opening_pinnacle_home_prob, last_fetched_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(game_date, home_team, away_team) DO UPDATE SET
                    home_odds                  = excluded.home_odds,
                    away_odds                  = excluded.away_odds,
                    pinnacle_home_odds         = excluded.pinnacle_home_odds,
                    pinnacle_away_odds         = excluded.pinnacle_away_odds,
                    best_home_odds             = excluded.best_home_odds,
                    best_away_odds             = excluded.best_away_odds,
                    best_home_book             = excluded.best_home_book,
                    best_away_book             = excluded.best_away_book,
                    consensus_home_prob        = excluded.consensus_home_prob,
                    opening_pinnacle_home_prob = COALESCE(
                        historical_odds.opening_pinnacle_home_prob,
                        excluded.opening_pinnacle_home_prob
                    ),
                    last_fetched_at = excluded.last_fetched_at
            """, (
                game_date, home_abbr, away_abbr,
                disp_h, disp_a,
                pin_h, pin_a,
                best_h_odds, best_a_odds, best_h_book, best_a_book,
                consensus, opening_pp, fetched_at,
            ))
            stored += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    print(f"  Stored odds for {stored} NBA game(s).")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    fetch_and_store_odds()
