import os
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import create_engine, text

router = APIRouter()

_HIGH_CONF_PROB  = 0.62   # max(home, away) >= this → HIGH tier
_MED_CONF_PROB   = 0.57   # max(home, away) >= this → MEDIUM tier


def _get_pg_engine():
    user = os.getenv("DB_USER", "postgres")
    pwd  = os.getenv("DB_PASS", "")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    name = os.getenv("DB_NAME", "mlb_model")
    return create_engine(f"postgresql://{user}:{pwd}@{host}:{port}/{name}")


def _load_predictions() -> pd.DataFrame:
    # Primary: PostgreSQL mlb_predictions table
    try:
        engine = _get_pg_engine()
        df = pd.read_sql(text("""
            SELECT
                game_date::text AS date,
                matchup, home_team, away_team,
                home_model_prob, away_model_prob, confidence, home_elo_prob,
                home_odds, away_odds, market_implied_home_prob,
                ev_home, ev_away, result
            FROM mlb_predictions
            ORDER BY game_date DESC, confidence DESC NULLS LAST
        """), engine)
        return df
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("MLB PostgreSQL unavailable (%s) — falling back to CSV", e)

    # Fallback: CSV
    path = os.getenv("BETS_LOG_PATH", "bets_log.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = df["date"].astype(str)
    return df


def _confidence_tier(home_prob: float, away_prob: float, mkt_home_prob: float | None) -> str:
    mx = max(home_prob, away_prob)
    if mx >= _HIGH_CONF_PROB:
        return "HIGH"
    if mx >= _MED_CONF_PROB:
        return "MEDIUM"
    return "LOW"


def _build_reason_text(row: pd.Series) -> str:
    home_prob  = float(row.get("home_model_prob", 0.5) or 0.5)
    away_prob  = float(row.get("away_model_prob", 0.5) or 0.5)
    home_elo   = float(row.get("home_elo_prob", 0.5)   or 0.5)
    home_team  = str(row.get("home_team", "Home"))
    away_team  = str(row.get("away_team", "Away"))
    ev_home    = row.get("ev_home")
    ev_away    = row.get("ev_away")

    import math as _math
    mkt_raw = row.get("market_implied_home_prob")
    try:
        _v = float(mkt_raw)
        mkt_home = _v if not _math.isnan(_v) else None
    except (TypeError, ValueError):
        mkt_home = None

    favored_is_home = home_prob >= away_prob
    favored_team    = home_team if favored_is_home else away_team
    fav_prob        = max(home_prob, away_prob)
    fav_elo         = home_elo if favored_is_home else (1 - home_elo)
    fav_mkt         = mkt_home if favored_is_home else (1 - mkt_home if mkt_home is not None else None)

    confidence_word = (
        "strongly" if fav_prob >= 0.65 else
        "moderately" if fav_prob >= 0.58 else
        "slightly"
    )

    # Why we picked them
    elo_gap = abs(fav_prob - fav_elo)
    if elo_gap <= 0.05:
        why = (
            f"We {confidence_word} favor {favored_team} to win. "
            f"Their season results back this up — both of our models agree."
        )
    elif fav_elo > fav_prob:
        why = (
            f"We {confidence_word} favor {favored_team} to win. "
            f"Their season results are even more in their favor than our main model suggests."
        )
    else:
        why = (
            f"We {confidence_word} favor {favored_team} to win, "
            f"though their season record is a bit less convincing — worth keeping in mind."
        )

    # Vegas context in plain English
    if fav_mkt is not None:
        edge = fav_prob - fav_mkt
        if edge > 0.08:
            why += f" Vegas has them at {fav_mkt*100:.0f}% — we think this game is more lopsided than the lines suggest."
        elif edge < -0.05:
            why += f" Vegas is even more confident in {favored_team} at {fav_mkt*100:.0f}%."
        else:
            why += f" Vegas agrees, putting {favored_team} at {fav_mkt*100:.0f}%."

    return why


def _enrich_predictions(df: pd.DataFrame) -> list[dict]:
    results = []
    for _, row in df.iterrows():
        home_prob = float(row.get("home_model_prob", 0.5) or 0.5)
        away_prob = float(row.get("away_model_prob", 0.5) or 0.5)
        import math as _m
        try:
            _mv = float(row.get("market_implied_home_prob"))
            mkt_home_prob = _mv if not _m.isnan(_mv) else None
        except (TypeError, ValueError):
            mkt_home_prob = None
        tier = _confidence_tier(home_prob, away_prob, mkt_home_prob)
        # Use the pre-computed confidence from the CSV if present, else derive it
        try:
            confidence = float(row["confidence"]) if "confidence" in row.index and pd.notna(row["confidence"]) else max(home_prob, away_prob)
        except (TypeError, ValueError):
            confidence = max(home_prob, away_prob)

        record = {
            "date":            str(row.get("date", "")),
            "matchup":         str(row.get("matchup", "")),
            "home_team":       str(row.get("home_team", "")),
            "away_team":       str(row.get("away_team", "")),
            "home_model_prob": round(home_prob, 4),
            "away_model_prob": round(away_prob, 4),
            "confidence":      round(confidence, 4),
            "home_elo_prob":   round(float(row.get("home_elo_prob", 0.5) or 0.5), 4),
            "result":          str(row.get("result", "TBD")),
            "confidence_tier": tier,
            "reason_text":     _build_reason_text(row),
        }
        # Pass through EV and market probability fields if present
        for col in ("home_odds", "away_odds", "ev_home", "ev_away", "market_implied_home_prob"):
            if col in row.index:
                try:
                    record[col] = round(float(row[col]), 4) if pd.notna(row[col]) else None
                except (TypeError, ValueError):
                    record[col] = None
        results.append(record)
    return results


# ── Predictions ───────────────────────────────────────────────────────────────

@router.get("/predictions")
def get_predictions(date: Optional[str] = Query(None)):
    """Return game predictions for a specific date with confidence tier and reason text."""
    df = _load_predictions()
    if df.empty:
        return {"date": None, "predictions": [], "total": 0}

    if date is None:
        date = df["date"].max()

    day = df[df["date"] == date].copy()
    if day.empty:
        raise HTTPException(status_code=404, detail=f"No predictions found for {date}.")

    for col in ["home_model_prob", "away_model_prob", "home_elo_prob"]:
        if col in day.columns:
            day[col] = pd.to_numeric(day[col], errors="coerce")

    # Sort by confidence descending — use pre-computed column when available
    day = day.copy()
    if "confidence" in day.columns:
        day["confidence"] = pd.to_numeric(day["confidence"], errors="coerce")
        day["_sort_conf"] = day["confidence"].fillna(day[["home_model_prob", "away_model_prob"]].max(axis=1))
    else:
        day["_sort_conf"] = day[["home_model_prob", "away_model_prob"]].max(axis=1)
    day = day.sort_values("_sort_conf", ascending=False).drop(columns="_sort_conf")

    return {
        "date":        date,
        "total":       len(day),
        "predictions": _enrich_predictions(day),
    }


@router.get("/history")
def get_history():
    """Return all historical predictions with accuracy stats."""
    df = _load_predictions()
    if df.empty:
        return {"predictions": [], "stats": None}

    for col in ["home_model_prob", "away_model_prob", "home_elo_prob"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    resolved = df[df["result"].isin(["HOME_WIN", "AWAY_WIN"])].copy()
    stats = None
    if not resolved.empty:
        resolved["predicted_home"] = resolved["home_model_prob"] > 0.5
        resolved["actual_home"]    = resolved["result"] == "HOME_WIN"
        correct = (resolved["predicted_home"] == resolved["actual_home"]).sum()
        stats = {
            "total_games": int(len(resolved)),
            "correct":     int(correct),
            "accuracy":    round(float(correct / len(resolved)), 4),
        }

    df = df.sort_values("date", ascending=False)
    return {
        "predictions": _enrich_predictions(df),
        "stats":       stats,
    }


@router.get("/dates")
def get_dates():
    df = _load_predictions()
    if df.empty:
        return {"dates": []}
    return {"dates": sorted(df["date"].unique().tolist(), reverse=True)[:30]}


# ── EV Bets ───────────────────────────────────────────────────────────────────

@router.get("/ev-bets")
def get_ev_bets(date: Optional[str] = Query(None)):
    """
    Return +EV bets for a given date (defaults to today / most recent).
    Includes Pinnacle vig-free probability, best available odds + book,
    line-movement direction, and CLV once the game is resolved.
    """
    try:
        engine = _get_pg_engine()
        df = pd.read_sql(text("""
            SELECT
                game_date::text AS date,
                matchup, side, team,
                model_prob, market_prob, pinnacle_prob, edge_vs_market,
                entry_odds, entry_book,
                ev, kelly_pct, line_move_direction,
                closing_pinnacle_odds, clv_pct,
                result
            FROM mlb_ev_bets
            ORDER BY game_date DESC, ev DESC
        """), engine)
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("mlb_ev_bets unavailable (%s)", e)
        return {"date": None, "total": 0, "bets": []}

    if df.empty:
        return {"date": None, "total": 0, "bets": []}

    if date is None:
        date = df["date"].max()

    day = df[df["date"] == date]
    if day.empty:
        return {"date": date, "total": 0, "bets": []}

    lm_label = {1: "confirming", -1: "against", 0: "neutral", None: None}

    bets = []
    for _, row in day.iterrows():
        def _f(col, digits=4):
            try:
                v = row.get(col)
                return round(float(v), digits) if v is not None and pd.notna(v) else None
            except (TypeError, ValueError):
                return None

        lm = row.get("line_move_direction")
        try:
            lm = int(lm) if lm is not None and pd.notna(lm) else 0
        except (TypeError, ValueError):
            lm = 0

        bets.append({
            "date":                 str(row["date"]),
            "matchup":              str(row["matchup"]),
            "side":                 str(row["side"]),
            "team":                 str(row["team"]),
            "model_prob":           _f("model_prob"),
            "market_prob":          _f("market_prob"),
            "pinnacle_prob":        _f("pinnacle_prob"),
            "edge_vs_market":       _f("edge_vs_market"),
            "entry_odds":           _f("entry_odds", 3),
            "entry_book":           str(row.get("entry_book") or ""),
            "ev":                   _f("ev", 2),
            "kelly_pct":            _f("kelly_pct", 2),
            "line_move_direction":  lm,
            "line_move_label":      lm_label.get(lm, "neutral"),
            "closing_pinnacle_odds": _f("closing_pinnacle_odds", 3),
            "clv_pct":              _f("clv_pct"),
            "result":               str(row.get("result", "TBD")),
        })

    return {"date": date, "total": len(bets), "bets": bets}


@router.get("/clv-summary")
def get_clv_summary():
    """
    Return rolling CLV statistics across all resolved EV bets.
    Used to track whether the model has genuine long-term edge.
    """
    try:
        engine = _get_pg_engine()
        df = pd.read_sql(text("""
            SELECT
                game_date::text AS date,
                side, team, matchup,
                entry_odds, closing_pinnacle_odds, clv_pct,
                ev, result
            FROM mlb_ev_bets
            WHERE clv_pct IS NOT NULL
            ORDER BY game_date DESC
        """), engine)
    except Exception:
        return {"total_bets": 0, "clv_positive_rate": None, "mean_clv_pct": None, "bets": []}

    if df.empty:
        return {"total_bets": 0, "clv_positive_rate": None, "mean_clv_pct": None, "bets": []}

    total    = len(df)
    pos_clv  = (df["clv_pct"] > 0).sum()
    mean_clv = float(df["clv_pct"].mean())

    bets = []
    for _, row in df.iterrows():
        bets.append({
            "date":                  str(row["date"]),
            "matchup":               str(row["matchup"]),
            "side":                  str(row["side"]),
            "team":                  str(row["team"]),
            "entry_odds":            round(float(row["entry_odds"]), 3) if pd.notna(row.get("entry_odds")) else None,
            "closing_pinnacle_odds": round(float(row["closing_pinnacle_odds"]), 3) if pd.notna(row.get("closing_pinnacle_odds")) else None,
            "clv_pct":               round(float(row["clv_pct"]), 4),
            "ev":                    round(float(row["ev"]), 2) if pd.notna(row.get("ev")) else None,
            "result":                str(row.get("result", "TBD")),
        })

    return {
        "total_bets":       total,
        "clv_positive_rate": round(float(pos_clv / total), 4) if total > 0 else None,
        "mean_clv_pct":     round(mean_clv, 4),
        "bets":             bets,
    }
