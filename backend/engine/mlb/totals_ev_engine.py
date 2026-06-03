"""
MLB Totals EV Engine v2.

Uses the half-game XGBoost model (totals_model.py):
  1. Predict home_runs and away_runs separately
  2. Sum for predicted total
  3. P(over/under) via Negative Binomial CDF with fitted alpha
  4. Edge vs Pinnacle vig-free probability, same gates as moneyline
"""

import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from scipy.stats import nbinom
from sqlalchemy import text

from db import get_engine
from feature_helpers import (
    build_bullpen_agg,
    build_umpire_k_lookup,
    get_bullpen_k_pct,
    get_rest_days,
    get_starter_rolling_stats,
    get_team_batting_advanced,
    get_team_record,
    get_team_rolling_stats,
    _LG_UMPIRE_K_RATE,
)
from totals_model import _GAME_SQL, _PITCHER_SQL, _UMPIRE_SQL
from utils import HALF_GAME_FEATURES, MLB_TEAM_MAP, PARK_FACTORS
from weather import LEAGUE_AVG_WEATHER, get_game_weather

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "xgb_mlb_totals.pkl")
STAKE           = 100.0
EV_THRESHOLD    = 5.0
MIN_MARKET_EDGE = 0.04
MAX_MARKET_EDGE = 0.07
LINE_MOVE_VETO  = 0.03


def _kelly_pct(prob: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b if b > 0 else 0.0
    return min(max(raw * 0.25, 0.0), 0.05) * 100.0


def _vig_free_over(over_odds: float, under_odds: float) -> float:
    raw_o = 1.0 / over_odds
    raw_u = 1.0 / under_odds
    return raw_o / (raw_o + raw_u)


def _p_over_negbin(mu: float, line: float, alpha: float) -> float:
    """P(actual total > line) using Negative Binomial with fitted dispersion alpha."""
    mu    = max(mu, 0.5)
    p     = alpha / (alpha + mu)
    floor_l = int(line)          # P(X > 8.5) = P(X >= 9) = P(X > 8)
    return float(nbinom.sf(floor_l, alpha, p))


def _ensure_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mlb_totals_ev_bets (
                id                    INTEGER PRIMARY KEY,
                game_date             TEXT NOT NULL,
                matchup               TEXT NOT NULL,
                side                  TEXT NOT NULL,
                label                 TEXT NOT NULL,
                total_line            REAL,
                predicted_total       REAL,
                model_prob            REAL,
                market_prob           REAL,
                pinnacle_prob         REAL,
                edge_vs_market        REAL,
                entry_odds            REAL,
                entry_book            TEXT,
                ev                    REAL,
                kelly_pct             REAL,
                line_move_direction   INTEGER DEFAULT 0,
                result                TEXT NOT NULL DEFAULT 'TBD',
                created_at            TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (game_date, matchup, side)
            )
        """))


def _fetch_probable_starters(date: str) -> dict:
    url = (f"https://statsapi.mlb.com/api/v1/schedule"
           f"?sportId=1&date={date}&hydrate=probablePitcher")
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: could not fetch probable starters — {e}")
        return {}
    starters = {}
    for date_block in resp.json().get("dates", []):
        for game in date_block.get("games", []):
            teams     = game.get("teams", {})
            home      = teams.get("home", {})
            away      = teams.get("away", {})
            home_abbr = MLB_TEAM_MAP.get(home.get("team", {}).get("name", ""), "")
            away_abbr = MLB_TEAM_MAP.get(away.get("team", {}).get("name", ""), "")
            home_sp   = home.get("probablePitcher", {}).get("id")
            away_sp   = away.get("probablePitcher", {}).get("id")
            if home_abbr and away_abbr:
                starters[(home_abbr, away_abbr)] = (home_sp, away_sp)
    return starters


def run_totals_predictions():
    print("--- MLB Totals EV Engine v2 ---")
    engine = get_engine()

    try:
        saved        = joblib.load(MODEL_PATH)
        model        = saved["xgb"]
        negbin_alpha = float(saved["negbin_alpha"])
    except FileNotFoundError:
        print("Totals model not found. Run totals_model.py first.")
        return []

    today = datetime.now().strftime("%Y-%m-%d")

    # Use earliest upcoming date in totals odds table
    try:
        with engine.connect() as c:
            row = c.execute(
                text("SELECT MIN(game_date) FROM historical_totals_odds WHERE game_date >= :today"),
                {"today": today}
            ).fetchone()
        target_date = row[0] if row and row[0] else today
    except Exception:
        target_date = today

    # Historical games for rolling features
    df_games = pd.read_sql(
        text(_GAME_SQL.replace("WHERE game_type = 'R'",
                               "WHERE game_type = 'R' AND game_date < :today")),
        engine, params={"today": today}, parse_dates=["game_date"]
    )
    df_games["home_win"] = (df_games["final_home_score"] > df_games["final_away_score"]).astype(int)
    df_games = df_games.dropna(subset=["home_xwoba", "away_xwoba"]).reset_index(drop=True)

    # Pitcher data
    df_pitcher = pd.read_sql(
        text(_PITCHER_SQL.replace("WHERE game_type = 'R'",
                                  "WHERE game_type = 'R' AND game_date < :today")),
        engine, params={"today": today}, parse_dates=["game_date"]
    )
    starter_idx = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    # Umpire lookup (rolling from history)
    umpire_lookup: dict = {}
    try:
        df_ump = pd.read_sql(
            text(_UMPIRE_SQL.replace("WHERE game_type='R' AND umpire IS NOT NULL",
                                     "WHERE game_type='R' AND umpire IS NOT NULL AND game_date < :today")),
            engine, params={"today": today}, parse_dates=["game_date"]
        )
        df_ump["game_date"] = df_ump["game_date"].astype(str).str[:10]
        umpire_lookup = build_umpire_k_lookup(df_ump)
    except Exception:
        pass

    # Totals odds for target date
    try:
        odds_df = pd.read_sql(text("""
            SELECT home_team, away_team, over_line,
                   pinnacle_over_odds, pinnacle_under_odds,
                   best_over_odds, best_over_book,
                   best_under_odds, best_under_book,
                   opening_pinnacle_over_odds
            FROM historical_totals_odds
            WHERE game_date = :target_date
        """), engine, params={"target_date": target_date})
    except Exception as e:
        print(f"  No totals odds found ({e}). Run fetch_odds.py first.")
        return []

    if odds_df.empty:
        print(f"  No totals odds for {target_date}.")
        return []

    totals_lookup = {(r["home_team"], r["away_team"]): dict(r) for _, r in odds_df.iterrows()}

    probable_starters = _fetch_probable_starters(target_date)
    if not probable_starters:
        print("  No games found.")
        return []

    weather_cache = {home: get_game_weather(home) for home, _ in probable_starters}
    today_ts = pd.Timestamp(target_date)

    ev_bets = []
    _ensure_table(engine)

    for (home, away), (home_sp_id, away_sp_id) in probable_starters.items():
        odict = totals_lookup.get((home, away))
        if not odict or odict.get("over_line") is None:
            continue

        # Rolling stats
        home_off = get_team_rolling_stats(home, df_games)
        away_off = get_team_rolling_stats(away, df_games)
        if not home_off or not away_off:
            continue
        home_bat = get_team_batting_advanced(home, df_games)
        away_bat = get_team_batting_advanced(away, df_games)
        if not home_bat or not away_bat:
            continue
        home_rec = get_team_record(home, df_games)
        away_rec = get_team_record(away, df_games)
        if not home_rec or not away_rec:
            continue
        home_sp = get_starter_rolling_stats(home_sp_id, df_starters)
        away_sp = get_starter_rolling_stats(away_sp_id, df_starters)
        if home_sp is None or away_sp is None:
            print(f"  Skipping {away} @ {home} — starter data unavailable.")
            continue
        home_bp = get_bullpen_k_pct(home, today_ts, df_bullpen_agg)
        away_bp = get_bullpen_k_pct(away, today_ts, df_bullpen_agg)
        if home_bp is None or away_bp is None:
            continue

        wx     = weather_cache.get(home, LEAGUE_AVG_WEATHER)
        park_f = PARK_FACTORS.get(home, 1.0)
        ump_k  = umpire_lookup.get((target_date, home), _LG_UMPIRE_K_RATE)

        common = {
            "home_park_factor":    park_f,
            "wind_component_out":  wx["wind_component_out"],
            "temperature_f":       wx["temperature_f"],
            "umpire_k_rate":       ump_k,
        }

        # HOME half: home offense vs away starter
        home_row = {
            **common,
            "off_xwoba":                  home_off["xwoba"],
            "off_rs_l15":                 home_rec["runs_scored"],
            "off_barrel_rate":            home_bat["barrel_rate"],
            "off_hard_hit_rate":          home_bat["hard_hit_rate"],
            "def_starter_k_pct":          away_sp["k_pct"],
            "def_starter_bb_pct":         away_sp["bb_pct"],
            "def_starter_xfip":           away_sp["xfip"],
            "def_starter_xwoba_against":  away_sp["xwoba_against"],
            "def_starter_gb_rate":        away_sp["gb_rate"],
            "def_bullpen_k_pct":          away_bp,
            "is_home":                    1.0,
            "team_rest_days":             get_rest_days(home, today_ts, df_games),
        }

        # AWAY half: away offense vs home starter
        away_row = {
            **common,
            "off_xwoba":                  away_off["xwoba"],
            "off_rs_l15":                 away_rec["runs_scored"],
            "off_barrel_rate":            away_bat["barrel_rate"],
            "off_hard_hit_rate":          away_bat["hard_hit_rate"],
            "def_starter_k_pct":          home_sp["k_pct"],
            "def_starter_bb_pct":         home_sp["bb_pct"],
            "def_starter_xfip":           home_sp["xfip"],
            "def_starter_xwoba_against":  home_sp["xwoba_against"],
            "def_starter_gb_rate":        home_sp["gb_rate"],
            "def_bullpen_k_pct":          home_bp,
            "is_home":                    0.0,
            "team_rest_days":             get_rest_days(away, today_ts, df_games),
        }

        feat_df = pd.DataFrame([home_row, away_row])[HALF_GAME_FEATURES]
        if feat_df.isnull().any().any():
            continue

        half_preds   = model.predict(feat_df)
        pred_total   = float(half_preds[0] + half_preds[1])

        line   = float(odict["over_line"])
        pin_o  = odict.get("pinnacle_over_odds")
        pin_u  = odict.get("pinnacle_under_odds")
        if not pin_o or not pin_u:
            continue

        pin_o, pin_u  = float(pin_o), float(pin_u)
        pin_over_prob = _vig_free_over(pin_o, pin_u)

        p_over  = _p_over_negbin(pred_total, line, negbin_alpha)
        p_under = 1.0 - p_over

        best_over  = float(odict["best_over_odds"])  if odict.get("best_over_odds")  else pin_o
        best_under = float(odict["best_under_odds"]) if odict.get("best_under_odds") else pin_u
        best_o_book = odict.get("best_over_book")  or "pinnacle"
        best_u_book = odict.get("best_under_book") or "pinnacle"

        # Line move direction
        opening_pin_o = odict.get("opening_pinnacle_over_odds")
        lm_over = lm_under = 0
        if opening_pin_o:
            delta = _vig_free_over(pin_o, pin_u) - _vig_free_over(float(opening_pin_o), pin_u)
            if abs(delta) >= 0.015:
                lm_over  =  1 if delta > 0 else -1
                lm_under = -lm_over

        matchup = f"{away} @ {home}"
        candidates = []
        for side, prob, mkt_prob, dec_odds, book, lm in [
            ("over",  p_over,  pin_over_prob,     best_over,  best_o_book, lm_over),
            ("under", p_under, 1-pin_over_prob,   best_under, best_u_book, lm_under),
        ]:
            edge = prob - mkt_prob
            if not (MIN_MARKET_EDGE <= edge <= MAX_MARKET_EDGE):
                continue
            ev = round((prob * (dec_odds - 1) * STAKE) - ((1 - prob) * STAKE), 2)
            if ev < EV_THRESHOLD:
                continue
            if lm == -1:
                print(f"  VETO {side} {matchup}: line moved against.")
                continue
            label = f"Over {line}" if side == "over" else f"Under {line}"
            candidates.append({
                "game_date":           target_date,
                "matchup":             matchup,
                "side":                side,
                "label":               label,
                "total_line":          line,
                "predicted_total":     round(pred_total, 2),
                "model_prob":          round(prob, 4),
                "market_prob":         round(mkt_prob, 4),
                "pinnacle_prob":       round(mkt_prob, 4),
                "edge_vs_market":      round(edge, 4),
                "entry_odds":          round(dec_odds, 3),
                "entry_book":          book,
                "ev":                  ev,
                "kelly_pct":           round(_kelly_pct(prob, dec_odds), 2),
                "line_move_direction": lm,
            })

        # One bet per game — highest EV side only
        if candidates:
            ev_bets.append(max(candidates, key=lambda x: x["ev"]))

    ev_bets.sort(key=lambda x: x["ev"], reverse=True)

    if ev_bets:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM mlb_totals_ev_bets WHERE game_date = :d"),
                         {"d": target_date})
            for b in ev_bets:
                conn.execute(text("""
                    INSERT INTO mlb_totals_ev_bets
                        (game_date, matchup, side, label, total_line, predicted_total,
                         model_prob, market_prob, pinnacle_prob, edge_vs_market,
                         entry_odds, entry_book, ev, kelly_pct, line_move_direction)
                    VALUES
                        (:game_date, :matchup, :side, :label, :total_line, :predicted_total,
                         :model_prob, :market_prob, :pinnacle_prob, :edge_vs_market,
                         :entry_odds, :entry_book, :ev, :kelly_pct, :line_move_direction)
                """), b)
        print(f"  {len(ev_bets)} totals EV bet(s) written.")
    else:
        print("  No +EV totals bets today.")

    lm_sym = {1: "↑", -1: "↓", 0: "—"}
    for b in ev_bets:
        print(
            f"  {b['matchup']:<20} {b['label']:<12} "
            f"pred {b['predicted_total']:.1f}  odds {b['entry_odds']:.3f} at {b['entry_book']:<10} "
            f"model {b['model_prob']*100:.1f}%  mkt {b['market_prob']*100:.1f}%  "
            f"edge +{b['edge_vs_market']*100:.1f}pp  EV ${b['ev']:.0f}  "
            f"{lm_sym.get(b['line_move_direction'], '—')}"
        )
    return ev_bets


if __name__ == "__main__":
    run_totals_predictions()
