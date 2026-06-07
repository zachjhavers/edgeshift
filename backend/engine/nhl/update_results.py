"""
Poll the NHL API for final scores, update the games table, and resolve CLV
for any EV bets whose games just finished.
"""

import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from db import get_conn
from utils import NHL_API_BASE

DB_PATH = Path(__file__).parent / "nhl_predictor.db"

_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": "edgeshift-nhl-engine/1.0"})


def _get(url: str) -> dict:
    try:
        resp = _SESSION.get(url, timeout=15)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return {}


def _pending_dates() -> list[str]:
    """Dates where either a game or an EV bet is still unresolved."""
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    conn = get_conn()
    game_dates = {r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM games WHERE result = 'TBD' AND game_date <= ?",
        (yesterday,),
    ).fetchall()}
    # Also include dates where EV bets are TBD but game is already resolved
    bet_dates = {r[0] for r in conn.execute(
        "SELECT DISTINCT game_date FROM nhl_ev_bets WHERE result = 'TBD' AND game_date <= ?",
        (yesterday,),
    ).fetchall()}
    conn.close()
    return sorted(game_dates | bet_dates)


def _update_from_schedule_data(data: dict) -> tuple[int, list[str]]:
    """Parse schedule API response. Returns (games_updated, resolved_dates)."""
    updates        = []
    resolved_dates = set()

    for day in data.get("gameWeek", []):
        game_date = day.get("date", "")
        for g in day.get("games", []):
            state = g.get("gameState", "")
            if state not in ("OFF", "FINAL"):
                continue
            game_id    = str(g.get("id", ""))
            home_score = (g.get("homeTeam") or {}).get("score")
            away_score = (g.get("awayTeam") or {}).get("score")
            if not game_id or home_score is None or away_score is None:
                continue
            result = "HOME_WIN" if home_score > away_score else "AWAY_WIN"
            updates.append((home_score, away_score, result, game_id))
            if game_date:
                resolved_dates.add(game_date)

    if not updates:
        return 0, []

    conn = get_conn()
    conn.executemany(
        "UPDATE games SET home_score=?, away_score=?, result=? WHERE game_id=? AND result='TBD'",
        updates,
    )
    changed = conn.total_changes
    conn.commit()
    conn.close()
    return changed, list(resolved_dates)


def _ensure_ev_cols():
    """Ensure nhl_ev_bets has all columns needed for CLV resolution."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nhl_ev_bets (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date              TEXT NOT NULL,
            matchup                TEXT NOT NULL,
            side                   TEXT NOT NULL,
            team                   TEXT NOT NULL,
            model_prob             REAL NOT NULL,
            market_prob            REAL NOT NULL,
            pinnacle_prob          REAL,
            edge_vs_market         REAL NOT NULL,
            odds                   REAL NOT NULL,
            entry_book             TEXT,
            ev                     REAL NOT NULL,
            kelly_pct              REAL NOT NULL,
            line_move_direction    INTEGER DEFAULT 0,
            closing_pinnacle_odds  REAL,
            clv_pct                REAL,
            result                 TEXT DEFAULT 'TBD',
            created_at             TEXT NOT NULL,
            UNIQUE(game_date, matchup, side)
        )
    """)
    conn.commit()
    for col, dtype in [
        ("pinnacle_prob",         "REAL"),
        ("entry_book",            "TEXT"),
        ("line_move_direction",   "INTEGER DEFAULT 0"),
        ("closing_pinnacle_odds", "REAL"),
        ("clv_pct",               "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE nhl_ev_bets ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass
    conn.close()


def _resolve_clv(resolved_dates: list[str]):
    """Update result + CLV for EV bets whose games just finished."""
    _ensure_ev_cols()
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row

    for date in resolved_dates:
        bets = conn.execute("""
            SELECT id, matchup, side, odds
            FROM nhl_ev_bets
            WHERE game_date = ?
        """, (date,)).fetchall()

        for bet in bets:
            parts = str(bet["matchup"]).split(" @ ")
            if len(parts) != 2:
                continue
            away_code, home_code = parts[0].strip(), parts[1].strip()

            game_row = conn.execute("""
                SELECT result FROM games
                WHERE game_date = ? AND home_team = ? AND away_team = ?
            """, (date, home_code, away_code)).fetchone()

            if game_row and game_row["result"] in ("HOME_WIN", "AWAY_WIN"):
                if bet["side"] == "home":
                    bet_result = "WIN" if game_row["result"] == "HOME_WIN" else "LOSS"
                else:
                    bet_result = "WIN" if game_row["result"] == "AWAY_WIN" else "LOSS"
                conn.execute(
                    "UPDATE nhl_ev_bets SET result = ? WHERE id = ? AND result = 'TBD'",
                    (bet_result, bet["id"]),
                )

            # CLV using closing Pinnacle odds
            odds_row = conn.execute("""
                SELECT pinnacle_home_odds, pinnacle_away_odds
                FROM historical_odds
                WHERE game_date = ? AND home_team = ? AND away_team = ?
            """, (date, home_code, away_code)).fetchone()

            if odds_row:
                closing = (
                    odds_row["pinnacle_home_odds"] if bet["side"] == "home"
                    else odds_row["pinnacle_away_odds"]
                )
                if closing and float(closing) > 0 and bet["odds"]:
                    clv_pct = round(float(bet["odds"]) / float(closing) - 1, 4)
                    conn.execute("""
                        UPDATE nhl_ev_bets
                        SET closing_pinnacle_odds = ?, clv_pct = ?
                        WHERE id = ? AND closing_pinnacle_odds IS NULL
                    """, (round(float(closing), 3), clv_pct, bet["id"]))

    conn.commit()
    conn.close()


def update_results():
    pending = _pending_dates()
    if not pending:
        print("No pending games to resolve.")
        return

    print(f"Checking results for {len(pending)} date(s): {pending[0]} → {pending[-1]}")
    total_updated = 0
    all_resolved  = []

    for date in pending:
        url  = f"{NHL_API_BASE}/v1/schedule/{date}"
        data = _get(url)
        n, resolved = _update_from_schedule_data(data)
        total_updated += n
        if n:
            print(f"  {date}: {n} game(s) resolved.")
            all_resolved.extend(resolved)
        time.sleep(0.2)

    if all_resolved:
        _resolve_clv(list(set(all_resolved)))
        print(f"  CLV resolved for {len(set(all_resolved))} date(s).")

    print(f"Done: {total_updated} game result(s) updated.")


if __name__ == "__main__":
    update_results()
