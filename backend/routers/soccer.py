import os
import sqlite3
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter()


def _conn():
    path = os.getenv("SOCCER_DB_PATH", "soccer_predictor.db")
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
            row  = conn.execute("SELECT MAX(match_date) AS d FROM soccer_ev_bets").fetchone()
            date = row["d"] if row and row["d"] else None

        if date is None:
            conn.close()
            return {"date": None, "bets": [], "total": 0}

        rows = conn.execute("""
            SELECT match_date, matchup, market, side, label,
                   model_prob, market_prob, pinnacle_prob, edge_vs_market,
                   entry_odds, entry_book, ev, kelly_pct,
                   line_move_direction, result
            FROM soccer_ev_bets
            WHERE match_date = ?
            ORDER BY ev DESC
        """, (date,)).fetchall()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"date": date, "bets": [], "total": 0}

    bets = []
    for r in rows:
        bets.append({
            "date":           r["match_date"],
            "matchup":        r["matchup"],
            "market":         r["market"],
            "side":           r["side"],
            "label":          r["label"],
            "model_prob":     _f(r["model_prob"]),
            "market_prob":    _f(r["market_prob"]),
            "pinnacle_prob":  _f(r["pinnacle_prob"]),
            "edge_vs_market": _f(r["edge_vs_market"]),
            "entry_odds":     _f(r["entry_odds"], 3),
            "entry_book":     r["entry_book"],
            "ev":             _f(r["ev"], 2),
            "kelly_pct":      _f(r["kelly_pct"], 2),
            "result":         r["result"],
        })

    return {"date": date, "bets": bets, "total": len(bets)}
