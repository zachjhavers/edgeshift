import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()


def _conn():
    path = os.getenv("NBA_DB_PATH", "nba_predictor.db")
    if not os.path.exists(path):
        return None
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _f(v, digits=4):
    try:
        return round(float(v), digits) if v is not None else None
    except (TypeError, ValueError):
        return None


@router.get("/ev-bets")
def get_ev_bets(date: Optional[str] = Query(None)):
    conn = _conn()
    if conn is None:
        return {"date": None, "bets": [], "total": 0}

    try:
        if date is None:
            from datetime import date as _date
            today = _date.today().isoformat()
            n = conn.execute("SELECT COUNT(*) FROM nba_ev_bets WHERE game_date = ?", (today,)).fetchone()[0]
            if n > 0:
                date = today
            else:
                row = conn.execute("SELECT MIN(game_date) FROM nba_ev_bets WHERE game_date >= ?", (today,)).fetchone()
                date = row[0] if row and row[0] else None

        if date is None:
            conn.close()
            return {"date": None, "bets": [], "total": 0}

        rows = conn.execute("""
            SELECT game_date, matchup, side, team, game_type,
                   model_prob, market_prob, pinnacle_prob, edge_vs_market,
                   odds, entry_book, ev, kelly_pct,
                   line_move_direction, closing_pinnacle_odds, clv_pct, result
            FROM nba_ev_bets
            WHERE game_date = ?
            ORDER BY ev DESC
        """, (date,)).fetchall()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"date": date, "bets": [], "total": 0}

    lm_label = {1: "confirming", -1: "against", 0: "neutral"}

    bets = []
    for r in rows:
        lm = int(r["line_move_direction"]) if r["line_move_direction"] is not None else 0

        book_raw   = r["entry_book"] or ""
        book_label = (book_raw
            .replace("williamhill_us", "Caesars")
            .replace("draftkings",     "DraftKings")
            .replace("fanduel",        "FanDuel")
            .replace("betmgm",         "BetMGM")
            .replace("pinnacle",       "Pinnacle"))

        bets.append({
            "date":                  str(r["game_date"]),
            "matchup":               str(r["matchup"]),
            "side":                  str(r["side"]),
            "team":                  str(r["team"]),
            "game_type":             str(r["game_type"]),
            "model_prob":            _f(r["model_prob"]),
            "market_prob":           _f(r["market_prob"]),
            "pinnacle_prob":         _f(r["pinnacle_prob"]),
            "edge_vs_market":        _f(r["edge_vs_market"]),
            "odds":                  _f(r["odds"], 3),
            "entry_book":            book_raw,
            "entry_book_label":      book_label,
            "ev":                    _f(r["ev"], 2),
            "kelly_pct":             _f(r["kelly_pct"], 2),
            "line_move_direction":   lm,
            "line_move_label":       lm_label.get(lm, "neutral"),
            "closing_pinnacle_odds": _f(r["closing_pinnacle_odds"], 3),
            "clv_pct":               _f(r["clv_pct"]),
            "result":                str(r["result"] or "TBD"),
        })

    return {"date": date, "total": len(bets), "bets": bets}
