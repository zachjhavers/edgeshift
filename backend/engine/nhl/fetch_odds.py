"""
Fetch NHL moneyline odds from The Odds API — Pinnacle + 4 US books in one call.
Computes consensus vig-free probability and stores Pinnacle odds separately.
Preserves opening_pinnacle_home_prob on first daily fetch (COALESCE on re-fetches).
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

import requests

ODDS_API_KEY = os.getenv("ODDS_API_KEY")
SPORT        = "icehockey_nhl"
REGIONS      = "us"
MARKETS      = "h2h"
BOOKMAKERS   = "pinnacle,draftkings,fanduel,betmgm,williamhill_us"

DB_PATH = Path(__file__).parent / "nhl_predictor.db"

# The Odds API returns full team names; NHL engine uses 3-letter codes
TEAM_NAME_MAP: dict[str, str] = {
    "Anaheim Ducks":         "ANA",
    "Boston Bruins":         "BOS",
    "Buffalo Sabres":        "BUF",
    "Calgary Flames":        "CGY",
    "Carolina Hurricanes":   "CAR",
    "Chicago Blackhawks":    "CHI",
    "Colorado Avalanche":    "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars":          "DAL",
    "Detroit Red Wings":     "DET",
    "Edmonton Oilers":       "EDM",
    "Florida Panthers":      "FLA",
    "Los Angeles Kings":     "LAK",
    "Minnesota Wild":        "MIN",
    "Montreal Canadiens":    "MTL",
    "Nashville Predators":   "NSH",
    "New Jersey Devils":     "NJD",
    "New York Islanders":    "NYI",
    "New York Rangers":      "NYR",
    "Ottawa Senators":       "OTT",
    "Philadelphia Flyers":   "PHI",
    "Pittsburgh Penguins":   "PIT",
    "Seattle Kraken":        "SEA",
    "San Jose Sharks":       "SJS",
    "St. Louis Blues":       "STL",
    "Tampa Bay Lightning":   "TBL",
    "Toronto Maple Leafs":   "TOR",
    "Utah Hockey Club":      "UTA",
    "Vancouver Canucks":     "VAN",
    "Vegas Golden Knights":  "VGK",
    "Washington Capitals":   "WSH",
    "Winnipeg Jets":         "WPG",
}


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    raw_h = 1.0 / home_odds
    raw_a = 1.0 / away_odds
    total = raw_h + raw_a
    return raw_h / total


def _ensure_odds_table():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS historical_odds (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date                  TEXT NOT NULL,
            home_team                  TEXT NOT NULL,
            away_team                  TEXT NOT NULL,
            home_odds                  REAL NOT NULL,
            away_odds                  REAL NOT NULL,
            bookmaker                  TEXT NOT NULL,
            fetched_at                 TEXT NOT NULL,
            pinnacle_home_odds         REAL,
            pinnacle_away_odds         REAL,
            best_home_odds             REAL,
            best_away_odds             REAL,
            best_home_book             TEXT,
            best_away_book             TEXT,
            consensus_home_prob        REAL,
            opening_pinnacle_home_prob REAL,
            UNIQUE(game_date, home_team, away_team)
        )
    """)
    conn.commit()

    new_cols = [
        ("pinnacle_home_odds",         "REAL"),
        ("pinnacle_away_odds",         "REAL"),
        ("best_home_odds",             "REAL"),
        ("best_away_odds",             "REAL"),
        ("best_home_book",             "TEXT"),
        ("best_away_book",             "TEXT"),
        ("consensus_home_prob",        "REAL"),
        ("opening_pinnacle_home_prob", "REAL"),
    ]
    for col, dtype in new_cols:
        try:
            conn.execute(f"ALTER TABLE historical_odds ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.close()


def fetch_and_store_odds():
    """Pull current NHL moneylines from multiple books and upsert into historical_odds."""
    if not ODDS_API_KEY:
        print("ODDS_API_KEY not set — skipping NHL odds fetch.")
        return

    print("Fetching NHL odds from The Odds API (multi-book)...")
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds/"
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
        print(f"  Failed to fetch NHL odds: {e}")
        return

    events     = resp.json()
    fetched_at = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    rows: list[tuple] = []

    for event in events:
        home_full = event["home_team"]
        away_full = event["away_team"]
        home_code = TEAM_NAME_MAP.get(home_full, home_full)
        away_code = TEAM_NAME_MAP.get(away_full, away_full)
        game_date = event["commence_time"].split("T")[0]

        book_odds: dict[str, tuple[float, float]] = {}
        for book in event.get("bookmakers", []):
            outcomes = book["markets"][0]["outcomes"]
            h = next((o["price"] for o in outcomes if o["name"] == home_full), None)
            a = next((o["price"] for o in outcomes if o["name"] == away_full), None)
            if h and a and float(h) > 1.0 and float(a) > 1.0:
                book_odds[book["key"]] = (float(h), float(a))

        if not book_odds:
            continue

        pin   = book_odds.get("pinnacle")
        pin_h = pin[0] if pin else None
        pin_a = pin[1] if pin else None

        vf_probs = [_vig_free_prob(h, a) for h, a in book_odds.values()]
        consensus_home_prob = round(sum(vf_probs) / len(vf_probs), 4)

        best_h_odds = max(h for h, _ in book_odds.values())
        best_h_book = next(k for k, (h, _) in book_odds.items() if h == best_h_odds)
        best_a_odds = max(a for _, a in book_odds.values())
        best_a_book = next(k for k, (_, a) in book_odds.items() if a == best_a_odds)

        # Legacy columns fall back to DraftKings, then first available book
        dk         = book_odds.get("draftkings")
        fallback_h = dk[0] if dk else list(book_odds.values())[0][0]
        fallback_a = dk[1] if dk else list(book_odds.values())[0][1]

        opening_pin_prob = round(_vig_free_prob(pin_h, pin_a), 4) if pin_h and pin_a else None

        rows.append((
            game_date, home_code, away_code,
            fallback_h, fallback_a, "multi", fetched_at,
            pin_h, pin_a,
            best_h_odds, best_a_odds, best_h_book, best_a_book,
            consensus_home_prob, opening_pin_prob,
        ))

    if not rows:
        print("  No valid NHL odds found.")
        return

    _ensure_odds_table()
    conn = _conn()
    conn.executemany("""
        INSERT INTO historical_odds
            (game_date, home_team, away_team, home_odds, away_odds, bookmaker, fetched_at,
             pinnacle_home_odds, pinnacle_away_odds,
             best_home_odds, best_away_odds, best_home_book, best_away_book,
             consensus_home_prob, opening_pinnacle_home_prob)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        ON CONFLICT(game_date, home_team, away_team) DO UPDATE SET
            home_odds                  = excluded.home_odds,
            away_odds                  = excluded.away_odds,
            bookmaker                  = excluded.bookmaker,
            fetched_at                 = excluded.fetched_at,
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
            )
    """, rows)
    conn.commit()
    conn.close()
    print(f"  {len(rows)} NHL odds record(s) stored.")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    fetch_and_store_odds()
