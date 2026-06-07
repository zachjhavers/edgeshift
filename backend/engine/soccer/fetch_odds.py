"""
Fetch FIFA World Cup odds from The Odds API.
Sport key: soccer_fifa_world_cup
Markets: h2h (1X2), totals (over/under goals)
"""

import os
import time
from datetime import datetime
from pathlib import Path

import requests

from db import get_conn
from utils import canonical

ODDS_API_KEY  = os.getenv("ODDS_API_KEY", "")
ODDS_API_BASE = "https://api.the-odds-api.com/v4/sports"
SPORT_KEY     = "soccer_fifa_world_cup"
SHARP_BOOK    = "pinnacle"
REGIONS       = "us,eu"
BOOKS         = ["pinnacle", "draftkings", "fanduel", "betmgm", "williamhill_us"]


def _remove_vig(odds: list[float]) -> list[float]:
    inv = [1 / o for o in odds if o > 1]
    total = sum(inv)
    return [i / total for i in inv]


def _best_book(outcomes: list[dict], side_name: str) -> tuple[float, str]:
    """Return (best decimal odds, book name) for a given outcome name."""
    best_odds = 0.0
    best_book = ""
    for book in outcomes:
        if book.get("name", "").lower() == SHARP_BOOK:
            continue
        for o in book.get("outcomes", []):
            if o.get("name", "") == side_name and float(o.get("price", 0)) > best_odds:
                best_odds = float(o["price"])
                best_book = book.get("key", "")
    return best_odds, best_book


def fetch_and_store_odds() -> int:
    if not ODDS_API_KEY:
        print("  No ODDS_API_KEY set — skipping odds fetch.")
        return 0

    print(f"Fetching FIFA World Cup odds (h2h + totals)...")

    stored = 0

    for market in ("h2h", "totals"):
        try:
            r = requests.get(
                f"{ODDS_API_BASE}/{SPORT_KEY}/odds/",
                params={
                    "apiKey":      ODDS_API_KEY,
                    "regions":     REGIONS,
                    "markets":     market,
                    "oddsFormat":  "decimal",
                    "bookmakers":  ",".join(BOOKS),
                },
                timeout=15,
            )
            r.raise_for_status()
            games = r.json()
            time.sleep(0.5)
        except Exception as e:
            print(f"  Odds API ({market}) failed: {e}")
            continue

        conn = get_conn()
        for game in games:
            home_team = canonical(game.get("home_team", ""))
            away_team = canonical(game.get("away_team", ""))
            commence  = game.get("commence_time", "")[:10]  # YYYY-MM-DD (UTC)

            if not home_team or not away_team:
                continue

            bms = game.get("bookmakers", [])
            pin = next((b for b in bms if b.get("key") == SHARP_BOOK), None)

            if market == "h2h":
                if not pin:
                    continue
                pin_outcomes = {o["name"]: float(o["price"])
                                for o in pin.get("markets", [{}])[0].get("outcomes", [])}
                pin_home = pin_outcomes.get(game.get("home_team", ""), 0)
                pin_draw = pin_outcomes.get("Draw", 0)
                pin_away = pin_outcomes.get(game.get("away_team", ""), 0)
                if not pin_home or not pin_draw or not pin_away:
                    continue

                try:
                    conn.execute("""
                        INSERT INTO historical_odds
                            (match_date, home_team, away_team,
                             pinnacle_home_odds, pinnacle_draw_odds, pinnacle_away_odds)
                        VALUES (?,?,?,?,?,?)
                        ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
                            pinnacle_home_odds = excluded.pinnacle_home_odds,
                            pinnacle_draw_odds = excluded.pinnacle_draw_odds,
                            pinnacle_away_odds = excluded.pinnacle_away_odds,
                            fetched_at         = datetime('now')
                    """, (commence, home_team, away_team, pin_home, pin_draw, pin_away))
                    stored += 1
                except Exception:
                    pass

            elif market == "totals":
                if not pin:
                    continue
                pin_markets = pin.get("markets", [])
                tot_market  = next((m for m in pin_markets if m.get("key") == "totals"), None)
                if not tot_market:
                    continue
                outcomes = tot_market.get("outcomes", [])
                over  = next((o for o in outcomes if o.get("name") == "Over"), None)
                under = next((o for o in outcomes if o.get("name") == "Under"), None)
                if not over or not under:
                    continue
                line      = float(over.get("point", 2.5))
                over_odds  = float(over["price"])
                under_odds = float(under["price"])

                try:
                    conn.execute("""
                        UPDATE historical_odds
                        SET pinnacle_over_odds  = ?,
                            pinnacle_under_odds = ?,
                            total_line          = ?,
                            fetched_at          = datetime('now')
                        WHERE match_date = ? AND home_team = ? AND away_team = ?
                    """, (over_odds, under_odds, line, commence, home_team, away_team))
                except Exception:
                    pass

        conn.commit()
        conn.close()

    print(f"  Stored odds for {stored} game(s).")
    return stored


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    from db import setup_db
    setup_db()
    fetch_and_store_odds()
