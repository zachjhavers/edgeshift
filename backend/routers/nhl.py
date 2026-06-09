import math
import os
import sqlite3
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

router = APIRouter()

_HIGH_CONF = 0.62
_MED_CONF  = 0.57


def _conn():
    path = os.getenv("NHL_DB_PATH", "nhl_predictor.db")
    if not os.path.exists(path):
        raise HTTPException(status_code=503, detail="NHL predictor DB not found.")
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_tier(home_prob: float, away_prob: float) -> str:
    mx = max(home_prob, away_prob)
    if mx >= _HIGH_CONF:
        return "HIGH"
    if mx >= _MED_CONF:
        return "MEDIUM"
    return "LOW"


def _build_reason_text(
    home_team: str,
    away_team: str,
    home_prob: float,
    away_prob: float,
    home_elo_prob: Optional[float],
    home_pp_pct: Optional[float],
    away_pp_pct: Optional[float],
    home_goalie_gsaa: Optional[float],
    away_goalie_gsaa: Optional[float],
    home_rest: Optional[int],
    away_rest: Optional[int],
) -> str:
    fav_is_home = home_prob >= away_prob
    fav         = home_team if fav_is_home else away_team
    fav_prob    = max(home_prob, away_prob)

    confidence_word = (
        "strongly" if fav_prob >= 0.65 else
        "moderately" if fav_prob >= 0.58 else
        "slightly"
    )

    text = f"We {confidence_word} favor {fav} to win."

    if home_pp_pct is not None and away_pp_pct is not None:
        fav_pp  = home_pp_pct if fav_is_home else away_pp_pct
        dog_pp  = away_pp_pct if fav_is_home else home_pp_pct
        pp_edge = fav_pp - dog_pp
        if pp_edge > 0.05:
            text += f" Their power play has been significantly sharper ({fav_pp*100:.0f}% vs {dog_pp*100:.0f}%)."
        elif pp_edge < -0.05:
            text += f" The opponent's power play has the edge — {fav} will need to stay out of the penalty box."

    if home_goalie_gsaa is not None and away_goalie_gsaa is not None:
        fav_gsaa = home_goalie_gsaa if fav_is_home else away_goalie_gsaa
        dog_gsaa = away_goalie_gsaa if fav_is_home else home_goalie_gsaa
        if fav_gsaa > 1.0 and fav_gsaa > dog_gsaa:
            text += f" Their goalie has been above average recently, saving an estimated {fav_gsaa:.1f} extra goals."
        elif dog_gsaa > 1.5 and dog_gsaa > fav_gsaa:
            text += f" The opposing goalie has been hot — {fav} will need strong puck luck to overcome that."

    if home_elo_prob is not None:
        fav_elo = home_elo_prob if fav_is_home else (1 - home_elo_prob)
        elo_gap = abs(fav_prob - fav_elo)
        if elo_gap <= 0.04:
            text += " Their season record backs this up."
        elif fav_elo > fav_prob + 0.05:
            text += " Their season-long results are even more in their favor than today's model suggests."

    if home_rest is not None and away_rest is not None:
        fav_rest = home_rest if fav_is_home else away_rest
        dog_rest = away_rest if fav_is_home else home_rest
        if fav_rest >= 2 and dog_rest == 1:
            text += f" Rest advantage: {fav} had {fav_rest} days off while the opponent is on a back-to-back."

    return text


def _load_predictions_df() -> pd.DataFrame:
    conn = _conn()
    df = pd.read_sql("""
        SELECT
            g.game_id,
            g.game_date AS date,
            g.home_team,
            g.away_team,
            g.result,
            g.game_type,
            p.home_win_prob,
            p.away_win_prob,
            f.home_elo_prob,
            f.home_pp_pct_10,
            f.away_pp_pct_10,
            f.home_goalie_gsaa_5,
            f.away_goalie_gsaa_5,
            f.home_rest_days,
            f.away_rest_days
        FROM predictions p
        JOIN games g ON g.game_id = p.game_id
        LEFT JOIN features f ON f.game_id = p.game_id
        ORDER BY g.game_date DESC, p.home_win_prob DESC
    """, conn)
    conn.close()
    return df


def _enrich(row: pd.Series) -> dict:
    home_prob = float(row.get("home_win_prob") or 0.5)
    away_prob = float(row.get("away_win_prob") or round(1 - home_prob, 4))
    result    = str(row.get("result") or "TBD")
    tier      = _confidence_tier(home_prob, away_prob)

    def _f(col) -> Optional[float]:
        v = row.get(col)
        if v is None:
            return None
        try:
            return float(v) if not math.isnan(float(v)) else None
        except (TypeError, ValueError):
            return None

    reason = _build_reason_text(
        home_team=str(row["home_team"]),
        away_team=str(row["away_team"]),
        home_prob=home_prob,
        away_prob=away_prob,
        home_elo_prob=_f("home_elo_prob"),
        home_pp_pct=_f("home_pp_pct_10"),
        away_pp_pct=_f("away_pp_pct_10"),
        home_goalie_gsaa=_f("home_goalie_gsaa_5"),
        away_goalie_gsaa=_f("away_goalie_gsaa_5"),
        home_rest=int(row["home_rest_days"]) if pd.notna(row.get("home_rest_days")) else None,
        away_rest=int(row["away_rest_days"]) if pd.notna(row.get("away_rest_days")) else None,
    )

    matchup = f"{row['away_team']} @ {row['home_team']}"

    return {
        "date":            str(row["date"]),
        "matchup":         matchup,
        "home_team":       str(row["home_team"]),
        "away_team":       str(row["away_team"]),
        "home_model_prob": round(home_prob, 4),
        "away_model_prob": round(away_prob, 4),
        "confidence":      round(max(home_prob, away_prob), 4),
        "home_elo_prob":   round(_f("home_elo_prob") or 0.5, 4),
        "result":          result,
        "confidence_tier": tier,
        "reason_text":     reason,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/predictions")
def get_predictions(date: Optional[str] = Query(None)):
    df = _load_predictions_df()
    if df.empty:
        return {"date": None, "predictions": [], "total": 0}

    if date is None:
        date = str(df["date"].max())

    day = df[df["date"] == date]
    if day.empty:
        raise HTTPException(status_code=404, detail=f"No predictions for {date}.")

    predictions = [_enrich(row) for _, row in day.iterrows()]
    predictions.sort(key=lambda x: -x["confidence"])

    return {"date": date, "total": len(predictions), "predictions": predictions}


@router.get("/history")
def get_history():
    df = _load_predictions_df()
    if df.empty:
        return {"predictions": [], "stats": None}

    predictions = [_enrich(row) for _, row in df.iterrows()]

    resolved = [p for p in predictions if p["result"] in ("HOME_WIN", "AWAY_WIN")]
    stats = None
    if resolved:
        correct = sum(
            1 for p in resolved
            if (p["home_model_prob"] >= 0.5 and p["result"] == "HOME_WIN") or
               (p["home_model_prob"] < 0.5  and p["result"] == "AWAY_WIN")
        )
        stats = {
            "total_games": len(resolved),
            "correct":     correct,
            "accuracy":    round(correct / len(resolved), 4),
        }

    return {"predictions": predictions, "stats": stats}


@router.get("/ev-bets")
def get_ev_bets(date: Optional[str] = Query(None)):
    """
    Return +EV bets for a given date (defaults to most recent).
    Includes Pinnacle vig-free probability, best available odds + book,
    line-movement direction, and CLV once the game is resolved.
    """
    path = os.getenv("NHL_DB_PATH", "nhl_predictor.db")
    if not os.path.exists(path):
        return {"date": None, "bets": [], "total": 0}

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        if date is None:
            from datetime import date as _date
            today = _date.today().isoformat()
            n = conn.execute("SELECT COUNT(*) FROM nhl_ev_bets WHERE game_date = ?", (today,)).fetchone()[0]
            if n > 0:
                date = today
            else:
                row = conn.execute("SELECT MIN(game_date) FROM nhl_ev_bets WHERE game_date > ?", (today,)).fetchone()
                date = row[0] if row and row[0] else None
            if date is None:
                row = conn.execute("SELECT MAX(game_date) FROM nhl_ev_bets").fetchone()
                date = row[0] if row and row[0] else None

        if date is None:
            conn.close()
            return {"date": None, "bets": [], "total": 0}

        rows = conn.execute("""
            SELECT game_date, matchup, side, team,
                   model_prob, market_prob, pinnacle_prob, edge_vs_market,
                   odds, entry_book, ev, kelly_pct,
                   line_move_direction, closing_pinnacle_odds, clv_pct, result
            FROM nhl_ev_bets
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

    def _f(v, digits=4):
        try:
            return round(float(v), digits) if v is not None else None
        except (TypeError, ValueError):
            return None

    bets = []
    for r in rows:
        lm = r["line_move_direction"]
        try:
            lm = int(lm) if lm is not None else 0
        except (TypeError, ValueError):
            lm = 0

        book_raw = r["entry_book"] or ""
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


@router.get("/clv-summary")
def get_clv_summary():
    """
    Return rolling CLV statistics across all resolved EV bets.
    Used to track whether the model has genuine long-term edge.
    """
    path = os.getenv("NHL_DB_PATH", "nhl_predictor.db")
    if not os.path.exists(path):
        return {"total_bets": 0, "clv_positive_rate": None, "mean_clv_pct": None, "bets": []}

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        rows = conn.execute("""
            SELECT game_date, matchup, side, team,
                   odds, closing_pinnacle_odds, clv_pct, ev, result
            FROM nhl_ev_bets
            WHERE clv_pct IS NOT NULL
            ORDER BY game_date DESC
        """).fetchall()
        conn.close()
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        return {"total_bets": 0, "clv_positive_rate": None, "mean_clv_pct": None, "bets": []}

    if not rows:
        return {"total_bets": 0, "clv_positive_rate": None, "mean_clv_pct": None, "bets": []}

    total    = len(rows)
    pos_clv  = sum(1 for r in rows if r["clv_pct"] is not None and float(r["clv_pct"]) > 0)
    mean_clv = sum(float(r["clv_pct"]) for r in rows) / total

    def _f(v, d=3):
        try:
            return round(float(v), d) if v is not None else None
        except (TypeError, ValueError):
            return None

    bets = [{
        "date":                  str(r["game_date"]),
        "matchup":               str(r["matchup"]),
        "side":                  str(r["side"]),
        "team":                  str(r["team"]),
        "odds":                  _f(r["odds"]),
        "closing_pinnacle_odds": _f(r["closing_pinnacle_odds"]),
        "clv_pct":               _f(r["clv_pct"], 4),
        "ev":                    _f(r["ev"], 2),
        "result":                str(r["result"] or "TBD"),
    } for r in rows]

    return {
        "total_bets":        total,
        "clv_positive_rate": round(float(pos_clv / total), 4) if total > 0 else None,
        "mean_clv_pct":      round(mean_clv, 4),
        "bets":              bets,
    }


@router.get("/dates")
def get_dates():
    df = _load_predictions_df()
    if df.empty:
        return {"dates": []}
    return {"dates": sorted(df["date"].unique().tolist(), reverse=True)[:30]}
