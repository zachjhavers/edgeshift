"""
MLB Totals EV Engine.

For each game with a stored O/U line:
  1. Build the same features used by totals_model.py
  2. Predict mean total runs via the XGBoost regressor
  3. Compute P(over) = norm.sf(line, predicted_mean, residual_std)
  4. Compare to Pinnacle vig-free over probability
  5. If edge 4-7pp and EV > $5/100, write to mlb_totals_ev_bets

Run standalone: python totals_ev_engine.py
"""

import os
import warnings
from datetime import datetime
from scipy.stats import norm

import joblib
import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from db import get_engine
from feature_helpers import (
    build_bullpen_agg,
    get_bullpen_k_pct,
    get_rest_days,
    get_starter_rolling_stats,
    get_team_record,
    get_team_rolling_stats,
)
from utils import MLB_TEAM_MAP, PARK_FACTORS, TOTALS_FEATURES
from weather import LEAGUE_AVG_WEATHER, get_game_weather

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH      = os.path.join(os.path.dirname(__file__), "xgb_mlb_totals.pkl")
STAKE           = 100.0
EV_THRESHOLD    = 5.0
MIN_MARKET_EDGE = 0.04
MAX_MARKET_EDGE = 0.07
LINE_MOVE_VETO  = 0.03


def _kelly_pct(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b if b > 0 else 0.0
    return min(max(raw * 0.25, 0.0), 0.05) * 100.0


def _vig_free_over(over_odds: float, under_odds: float) -> float:
    raw_o = 1.0 / over_odds
    raw_u = 1.0 / under_odds
    return raw_o / (raw_o + raw_u)


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
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date}&hydrate=probablePitcher"
    )
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
    print("--- MLB Totals EV Engine ---")
    engine = get_engine()

    try:
        saved        = joblib.load(MODEL_PATH)
        model        = saved["xgb"]
        residual_std = float(saved["residual_std"])
    except FileNotFoundError:
        print("Totals model not found. Run totals_model.py first.")
        return []

    today = datetime.now().strftime("%Y-%m-%d")

    # Use the earliest upcoming date in the totals odds table (handles UTC crossover)
    try:
        target_row = engine.connect().execute(
            text("SELECT MIN(game_date) FROM historical_totals_odds WHERE game_date >= :today"),
            {"today": today}
        ).fetchone()
        target_date = target_row[0] if target_row and target_row[0] else today
    except Exception:
        target_date = today

    # Load game data for feature building
    df_games = pd.read_sql(text("""
        SELECT game_pk, game_date, home_team, away_team,
               MAX(home_score) AS final_home_score,
               MAX(away_score) AS final_away_score,
               AVG(CASE WHEN inning_topbot='Top' THEN release_speed END)                   AS home_pitch_velo,
               AVG(CASE WHEN inning_topbot='Bot' THEN release_speed END)                   AS away_pitch_velo,
               AVG(CASE WHEN inning_topbot='Bot' THEN launch_speed END)                    AS home_bat_exit_velo,
               AVG(CASE WHEN inning_topbot='Top' THEN launch_speed END)                    AS away_bat_exit_velo,
               AVG(CASE WHEN inning_topbot='Bot' THEN estimated_woba_using_speedangle END) AS home_xwoba,
               AVG(CASE WHEN inning_topbot='Top' THEN estimated_woba_using_speedangle END) AS away_xwoba
        FROM statcast_raw
        WHERE game_type='R' AND game_date < :today
        GROUP BY game_pk, game_date, home_team, away_team
        HAVING MAX(home_score) IS NOT NULL
        ORDER BY game_date
    """), engine, params={"today": today}, parse_dates=["game_date"])
    df_games["home_win"] = (df_games["final_home_score"] > df_games["final_away_score"]).astype(int)

    pitcher_q = text("""
        SELECT game_pk, game_date, home_team, away_team, pitcher, inning_topbot,
               AVG(release_speed) AS avg_velo,
               CAST(SUM(CASE WHEN events='strikeout' THEN 1 ELSE 0 END) AS REAL) /
                   NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS k_pct,
               CAST(SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS REAL) /
                   NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS bb_pct,
               AVG(estimated_woba_using_speedangle) AS xwoba_against,
               SUM(CASE WHEN events='home_run' THEN 1 ELSE 0 END) AS hr_count,
               SUM(CASE WHEN events='strikeout' THEN 1 ELSE 0 END) AS k_count,
               SUM(CASE WHEN events IN ('walk','intent_walk') THEN 1 ELSE 0 END) AS bb_count,
               COUNT(*) AS pitch_count,
               SUM(CASE
                   WHEN events IN ('strikeout','field_out','force_out','sac_bunt','sac_fly',
                                   'fielders_choice_out','other_out','caught_stealing_2b',
                                   'caught_stealing_3b','caught_stealing_home',
                                   'pickoff_caught_stealing_2b','pickoff_caught_stealing_3b',
                                   'pickoff_caught_stealing_home','sac_bunt_double_play') THEN 1
                   WHEN events IN ('grounded_into_double_play','strikeout_double_play',
                                   'double_play','sac_fly_double_play') THEN 2
                   WHEN events = 'triple_play' THEN 3
                   ELSE 0 END) * 1.0 / 3.0 AS ip
        FROM statcast_raw WHERE game_type='R' AND game_date < :today
        GROUP BY game_pk, game_date, home_team, away_team, pitcher, inning_topbot
    """)
    df_pitcher  = pd.read_sql(pitcher_q, engine, params={"today": today}, parse_dates=["game_date"])
    starter_idx = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    # Today's totals odds
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
        print("  No totals odds for today.")
        return []

    totals_lookup = {}
    for _, r in odds_df.iterrows():
        totals_lookup[(r["home_team"], r["away_team"])] = dict(r)

    probable_starters = _fetch_probable_starters(target_date)
    if not probable_starters:
        print("  No games found.")
        return []

    weather_cache = {}
    for home, _ in probable_starters:
        weather_cache[home] = get_game_weather(home)

    today_ts = pd.Timestamp(today)
    rows = []
    for (home, away), (home_sp_id, away_sp_id) in probable_starters.items():
        odict = totals_lookup.get((home, away))
        if not odict or odict.get("over_line") is None:
            continue

        team_h = get_team_rolling_stats(home, df_games)
        team_a = get_team_rolling_stats(away, df_games)
        if team_h is None or team_a is None:
            continue
        rec_h = get_team_record(home, df_games)
        rec_a = get_team_record(away, df_games)
        if rec_h is None or rec_a is None:
            continue
        sp_h = get_starter_rolling_stats(home_sp_id, df_starters)
        sp_a = get_starter_rolling_stats(away_sp_id, df_starters)
        if sp_h is None or sp_a is None:
            print(f"  Skipping {away} @ {home} — starter data unavailable.")
            continue
        bp_h = get_bullpen_k_pct(home, today_ts, df_bullpen_agg)
        bp_a = get_bullpen_k_pct(away, today_ts, df_bullpen_agg)
        if bp_h is None or bp_a is None:
            continue

        rest_h = get_rest_days(home, today_ts, df_games)
        rest_a = get_rest_days(away, today_ts, df_games)
        wx     = weather_cache.get(home, LEAGUE_AVG_WEATHER)

        rows.append({
            "home_team": home,
            "away_team": away,
            "home_rs_l15":                rec_h["runs_scored"],
            "away_rs_l15":                rec_a["runs_scored"],
            "home_ra_l15":                rec_h["runs_allowed"],
            "away_ra_l15":                rec_a["runs_allowed"],
            "home_xwoba":                 team_h["xwoba"],
            "away_xwoba":                 team_a["xwoba"],
            "home_starter_k_pct":         sp_h["k_pct"],
            "away_starter_k_pct":         sp_a["k_pct"],
            "home_starter_bb_pct":        sp_h["bb_pct"],
            "away_starter_bb_pct":        sp_a["bb_pct"],
            "home_starter_fip":           sp_h["fip"],
            "away_starter_fip":           sp_a["fip"],
            "home_starter_xwoba_against": sp_h["xwoba_against"],
            "away_starter_xwoba_against": sp_a["xwoba_against"],
            "home_starter_ip":            sp_h["ip"],
            "away_starter_ip":            sp_a["ip"],
            "home_bullpen_k_pct":         bp_h,
            "away_bullpen_k_pct":         bp_a,
            "home_park_factor":           PARK_FACTORS.get(home, 1.0),
            "wind_component_out":         wx["wind_component_out"],
            "wind_speed_mph":             wx["wind_speed_mph"],
            "temperature_f":              wx["temperature_f"],
            "home_rest_days":             rest_h,
            "away_rest_days":             rest_a,
        })

    if not rows:
        print("  Could not build features for any game.")
        return []

    df = pd.DataFrame(rows).dropna(subset=TOTALS_FEATURES)
    if df.empty:
        print("  All games dropped after NaN filter.")
        return []

    predicted_totals = model.predict(df[TOTALS_FEATURES])
    df["predicted_total"] = predicted_totals

    ev_bets = []
    _ensure_table(engine)

    for i, row in df.iterrows():
        home  = row["home_team"]
        away  = row["away_team"]
        pred  = float(row["predicted_total"])
        odict = totals_lookup.get((home, away))
        if not odict:
            continue

        line = float(odict["over_line"])
        pin_o = odict.get("pinnacle_over_odds")
        pin_u = odict.get("pinnacle_under_odds")

        if not pin_o or not pin_u:
            continue

        pin_o, pin_u = float(pin_o), float(pin_u)
        pin_over_prob = _vig_free_over(pin_o, pin_u)

        # P(over) from our regression: normal CDF with half-integer adjustment
        p_over  = float(norm.sf(line, pred, residual_std))
        p_under = 1.0 - p_over

        best_over  = float(odict["best_over_odds"])  if odict.get("best_over_odds")  else pin_o
        best_under = float(odict["best_under_odds"]) if odict.get("best_under_odds") else pin_u
        best_o_book = odict.get("best_over_book")  or "pinnacle"
        best_u_book = odict.get("best_under_book") or "pinnacle"

        # Line move direction (vs opening Pinnacle)
        opening_pin_o = odict.get("opening_pinnacle_over_odds")
        lm_dir = 0
        if opening_pin_o and pin_o:
            current_pp = _vig_free_over(float(pin_o), float(pin_u))
            opening_pp = _vig_free_over(float(opening_pin_o), float(pin_u))
            delta_over = current_pp - opening_pp
            if abs(delta_over) >= 0.015:
                lm_dir_over  = 1 if delta_over > 0 else -1
                lm_dir_under = -lm_dir_over

        matchup = f"{away} @ {home}"

        for side, prob, mkt_prob, dec_odds, book, lm in [
            ("over",  p_over,  pin_over_prob,       best_over,  best_o_book, lm_dir if lm_dir != 0 else 0),
            ("under", p_under, 1 - pin_over_prob,   best_under, best_u_book, -lm_dir if lm_dir != 0 else 0),
        ]:
            edge = prob - mkt_prob
            if not (MIN_MARKET_EDGE <= edge <= MAX_MARKET_EDGE):
                continue

            ev = round((prob * (dec_odds - 1) * STAKE) - ((1 - prob) * STAKE), 2)
            if ev < EV_THRESHOLD:
                continue

            # Line move veto
            if lm == -1:
                print(f"  VETO {side} {matchup}: line moved against.")
                continue

            label = f"Over {line}" if side == "over" else f"Under {line}"
            ev_bets.append({
                "game_date":          target_date,
                "matchup":            matchup,
                "side":               side,
                "label":              label,
                "total_line":         line,
                "predicted_total":    round(pred, 2),
                "model_prob":         round(prob, 4),
                "market_prob":        round(mkt_prob, 4),
                "pinnacle_prob":      round(mkt_prob, 4),
                "edge_vs_market":     round(edge, 4),
                "entry_odds":         round(dec_odds, 3),
                "entry_book":         book,
                "ev":                 ev,
                "kelly_pct":          round(_kelly_pct(prob, dec_odds), 2),
                "line_move_direction": lm,
            })

    # One bet per game — keep the higher-EV side only
    seen: dict[str, dict] = {}
    for b in ev_bets:
        key = b["matchup"]
        if key not in seen or b["ev"] > seen[key]["ev"]:
            seen[key] = b
    ev_bets = sorted(seen.values(), key=lambda x: x["ev"], reverse=True)

    if ev_bets:
        with engine.begin() as conn:
            # Clear today's entries first so stale over/under from prior runs don't linger
            conn.execute(text(
                "DELETE FROM mlb_totals_ev_bets WHERE game_date = :d"
            ), {"d": target_date})
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
                    ON CONFLICT (game_date, matchup, side) DO UPDATE SET
                        label               = EXCLUDED.label,
                        total_line          = EXCLUDED.total_line,
                        predicted_total     = EXCLUDED.predicted_total,
                        model_prob          = EXCLUDED.model_prob,
                        market_prob         = EXCLUDED.market_prob,
                        pinnacle_prob       = EXCLUDED.pinnacle_prob,
                        edge_vs_market      = EXCLUDED.edge_vs_market,
                        entry_odds          = EXCLUDED.entry_odds,
                        entry_book          = EXCLUDED.entry_book,
                        ev                  = EXCLUDED.ev,
                        kelly_pct           = EXCLUDED.kelly_pct,
                        line_move_direction = EXCLUDED.line_move_direction,
                        updated_at          = datetime('now')
                """), b)
        print(f"  {len(ev_bets)} totals EV bet(s) written.")
    else:
        print("  No +EV totals bets today.")

    lm_sym = {1: "↑", -1: "↓", 0: "—"}
    for b in ev_bets:
        print(
            f"  {b['matchup']:<20} {b['label']:<12} "
            f"odds {b['entry_odds']:.3f} at {b['entry_book']:<10} "
            f"model {b['model_prob']*100:.1f}%  mkt {b['market_prob']*100:.1f}%  "
            f"edge +{b['edge_vs_market']*100:.1f}pp  EV ${b['ev']:.0f}  "
            f"{lm_sym.get(b['line_move_direction'], '—')}"
        )

    return ev_bets


if __name__ == "__main__":
    run_totals_predictions()
