"""
Daily prediction + EV engine.

Changes from v1:
- Market baseline: Pinnacle vig-free prob (falls back to consensus, then DK)
- EV calculated against best available odds across all books (not just DK)
- Line movement filter: skip if closing line moved ≥3pp against our side
- MAX_MARKET_EDGE tightened to 7pp (from 10pp) per anti-calibration research
- Writes +EV bets to mlb_ev_bets PostgreSQL table for CLV tracking
- Weather features from Open-Meteo via weather.py
- Opponent xwOBA quality features

Run standalone: python ev_engine.py
"""

import csv
import os
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import requests
from sqlalchemy import text

from db import get_engine
from feature_helpers import (
    ELO_HOME_ADV,
    ELO_INIT,
    build_bullpen_agg,
    compute_current_elo,
    get_bullpen_k_pct,
    get_opponent_xwoba,
    get_rest_days,
    get_starter_rolling_stats,
    get_team_record,
    get_team_rolling_stats,
)
from utils import FEATURES, MLB_TEAM_MAP, PARK_FACTORS
from weather import LEAGUE_AVG_WEATHER, get_game_weather

warnings.filterwarnings("ignore", category=UserWarning)

MODEL_PATH       = os.path.join(os.path.dirname(__file__), "xgb_mlb_v1.pkl")
PREDICTIONS_PATH = os.path.join(os.path.dirname(__file__), "bets_log.csv")
EV_BETS_PATH     = os.path.join(os.path.dirname(__file__), "ev_bets.csv")

STAKE           = 100.0
EV_THRESHOLD    = 15.0   # minimum EV per $100 staked to log a bet

# Pinnacle vig-free edge gates.
# Tightened MAX to 7pp (from 10pp) — published research shows the market wins
# ~77% of disagreements at ≥20pp and meaningful anti-calibration starts around 10pp.
MIN_MARKET_EDGE = 0.04
MAX_MARKET_EDGE = 0.07

# If the closing line (updated pre-game odds fetch) moved this many pp or more
# AGAINST our predicted side, skip — the market has new information we don't.
LINE_MOVE_VETO_PP = 0.03


def _kelly_pct(prob: float, decimal_odds: float) -> float:
    """Quarter-Kelly fraction as a percentage of bankroll, capped at 5%."""
    b = decimal_odds - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b if b > 0 else 0.0
    return min(max(raw * 0.25, 0.0), 0.05) * 100.0


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    """Return vig-removed implied home win probability from decimal odds."""
    raw_home = 1.0 / home_odds
    raw_away = 1.0 / away_odds
    return raw_home / (raw_home + raw_away)


def _ensure_ev_bets_table(engine) -> None:
    """Create mlb_ev_bets table if it doesn't exist."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mlb_ev_bets (
                id                    INTEGER PRIMARY KEY,
                game_date             TEXT NOT NULL,
                matchup               TEXT NOT NULL,
                side                  TEXT NOT NULL,
                team                  TEXT NOT NULL,
                model_prob            REAL,
                market_prob           REAL,
                pinnacle_prob         REAL,
                edge_vs_market        REAL,
                entry_odds            REAL,
                entry_book            TEXT,
                ev                    REAL,
                kelly_pct             REAL,
                line_move_direction   INTEGER,
                closing_pinnacle_odds REAL,
                clv_pct               REAL,
                result                TEXT NOT NULL DEFAULT 'TBD',
                created_at            TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at            TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE (game_date, matchup, side)
            )
        """))


def fetch_probable_starters(date: str) -> dict:
    """Return {(home_abbr, away_abbr): (home_sp_id, away_sp_id)} for today's games."""
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date}&hydrate=probablePitcher"
    )
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0",
            "Accept":     "application/json",
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


def run_predictions():
    """
    Build features for today's games and run the moneyline model.
    Writes results to bets_log.csv and mlb_ev_bets PostgreSQL table.
    Returns a list of prediction dicts.
    """
    print("--- Running MLB Prediction Engine (v2) ---")
    engine = get_engine()

    try:
        saved     = joblib.load(MODEL_PATH)
        model_raw = saved["xgb"]
    except FileNotFoundError:
        print("Model file not found. Run model_builder.py first.")
        return []

    calibrator = saved.get("platt") or saved.get("iso")
    if calibrator is None:
        print("Model file is missing calibrator. Run model_builder.py to retrain.")
        return []

    def predict_prob(X):
        raw = model_raw.predict_proba(X)[:, 1].reshape(-1, 1)
        if hasattr(calibrator, "predict_proba"):
            return calibrator.predict_proba(raw)[:, 1]
        return calibrator.transform(raw.ravel())

    today = datetime.now().strftime("%Y-%m-%d")

    print("Fetching today's games from MLB Stats API...")
    probable_starters = fetch_probable_starters(today)
    if not probable_starters:
        print("No games found for today.")
        return []

    game_stats_query = text("""
        SELECT
            game_date,
            home_team, away_team,
            MAX(home_score)  AS final_home_score,
            MAX(away_score)  AS final_away_score,
            AVG(CASE WHEN inning_topbot = 'Top' THEN release_speed                   END) AS home_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN release_speed                   END) AS away_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN launch_speed                    END) AS home_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Top' THEN launch_speed                    END) AS away_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN estimated_woba_using_speedangle END) AS home_xwoba,
            AVG(CASE WHEN inning_topbot = 'Top' THEN estimated_woba_using_speedangle END) AS away_xwoba
        FROM statcast_raw
        WHERE game_type = 'R'
          AND game_date < :today
        GROUP BY game_date, home_team, away_team
        HAVING MAX(home_score) IS NOT NULL
        ORDER BY game_date
    """)
    df_games = pd.read_sql(game_stats_query, engine, params={"today": today},
                           parse_dates=["game_date"])
    df_games["home_win"] = (df_games["final_home_score"] > df_games["final_away_score"]).astype(int)

    print("Computing current Elo ratings...")
    current_elo = compute_current_elo(df_games)

    pitcher_query = text("""
        SELECT
            game_pk,
            game_date,
            home_team, away_team,
            pitcher, inning_topbot,
            AVG(release_speed)      AS avg_velo,
            CAST(SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END) AS REAL) /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS k_pct,
            CAST(SUM(CASE WHEN events IN ('walk', 'intent_walk') THEN 1 ELSE 0 END) AS REAL) /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS bb_pct,
            AVG(estimated_woba_using_speedangle) AS xwoba_against,
            SUM(CASE WHEN events = 'home_run'                             THEN 1 ELSE 0 END) AS hr_count,
            SUM(CASE WHEN events = 'strikeout'                            THEN 1 ELSE 0 END) AS k_count,
            SUM(CASE WHEN events IN ('walk', 'intent_walk')              THEN 1 ELSE 0 END) AS bb_count,
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
        FROM statcast_raw
        WHERE game_type = 'R'
          AND game_date < :today
        GROUP BY game_pk, game_date, home_team, away_team, pitcher, inning_topbot
    """)
    df_pitcher  = pd.read_sql(pitcher_query, engine, params={"today": today},
                               parse_dates=["game_date"])
    starter_idx = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_pitcher["is_starter"] = df_pitcher["is_starter"].astype(bool)
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    # Load today's multi-book odds
    print("Loading today's odds (multi-book) from database...")
    odds_lookup: dict[tuple, dict] = {}
    try:
        odds_df = pd.read_sql(
            text("""
                SELECT home_team, away_team,
                       home_odds, away_odds,
                       pinnacle_home_odds, pinnacle_away_odds,
                       consensus_home_prob,
                       best_home_odds, best_away_odds,
                       best_home_book, best_away_book,
                       opening_pinnacle_home_prob
                FROM historical_odds
                WHERE game_date = :today
            """),
            engine, params={"today": today},
        )
        for _, r in odds_df.iterrows():
            odds_lookup[(r["home_team"], r["away_team"])] = {
                "home_odds":               float(r["home_odds"]),
                "away_odds":               float(r["away_odds"]),
                "pinnacle_home_odds":      float(r["pinnacle_home_odds"]) if pd.notna(r.get("pinnacle_home_odds")) else None,
                "pinnacle_away_odds":      float(r["pinnacle_away_odds"]) if pd.notna(r.get("pinnacle_away_odds")) else None,
                "consensus_home_prob":     float(r["consensus_home_prob"]) if pd.notna(r.get("consensus_home_prob")) else None,
                "best_home_odds":          float(r["best_home_odds"]) if pd.notna(r.get("best_home_odds")) else None,
                "best_away_odds":          float(r["best_away_odds"]) if pd.notna(r.get("best_away_odds")) else None,
                "best_home_book":          r.get("best_home_book"),
                "best_away_book":          r.get("best_away_book"),
                "opening_pinnacle_home_prob": float(r["opening_pinnacle_home_prob"]) if pd.notna(r.get("opening_pinnacle_home_prob")) else None,
            }
        print(f"  {len(odds_lookup)} game(s) with odds loaded.")
    except Exception as e:
        print(f"  Warning: could not load odds ({e}). EV calculations will be skipped.")

    # Prefetch weather for all home teams in today's slate
    print("Fetching game-time weather conditions...")
    weather_cache: dict[str, dict] = {}
    unique_home_teams = {h for h, _ in probable_starters}
    for home in unique_home_teams:
        weather_cache[home] = get_game_weather(home)

    rows = []
    for (home, away), (home_sp_id, away_sp_id) in probable_starters.items():
        team_h = get_team_rolling_stats(home, df_games)
        team_a = get_team_rolling_stats(away, df_games)
        if team_h is None or team_a is None:
            print(f"  Skipping {away} @ {home} — insufficient team history.")
            continue

        rec_h = get_team_record(home, df_games)
        rec_a = get_team_record(away, df_games)
        if rec_h is None or rec_a is None:
            print(f"  Skipping {away} @ {home} — insufficient record history.")
            continue

        sp_h = get_starter_rolling_stats(home_sp_id, df_starters)
        sp_a = get_starter_rolling_stats(away_sp_id, df_starters)
        if sp_h is None or sp_a is None:
            print(f"  Skipping {away} @ {home} — starter data unavailable.")
            continue

        elo_h    = current_elo.get(home, ELO_INIT)
        elo_a    = current_elo.get(away, ELO_INIT)
        elo_prob = 1 / (1 + 10 ** ((elo_a - elo_h - ELO_HOME_ADV) / 400))

        today_ts = pd.Timestamp(today)
        bp_h = get_bullpen_k_pct(home, today_ts, df_bullpen_agg)
        bp_a = get_bullpen_k_pct(away, today_ts, df_bullpen_agg)
        if bp_h is None or bp_a is None:
            print(f"  Skipping {away} @ {home} — insufficient bullpen history.")
            continue

        rest_h = get_rest_days(home, today_ts, df_games)
        rest_a = get_rest_days(away, today_ts, df_games)

        # Opponent xwOBA: quality of offenses each team's starters have faced recently
        opp_xwoba_h = get_opponent_xwoba(home, df_games) or 0.320
        opp_xwoba_a = get_opponent_xwoba(away, df_games) or 0.320

        # Weather at home stadium
        wx = weather_cache.get(home, LEAGUE_AVG_WEATHER)

        rows.append({
            "home_team":                  home,
            "away_team":                  away,
            "home_pitch_velo":            team_h["pitch_velo"],
            "away_pitch_velo":            team_a["pitch_velo"],
            "home_bat_exit_velo":         team_h["bat_exit_velo"],
            "away_bat_exit_velo":         team_a["bat_exit_velo"],
            "home_xwoba":                 team_h["xwoba"],
            "away_xwoba":                 team_a["xwoba"],
            "pitch_velo_diff":            team_h["pitch_velo"]    - team_a["pitch_velo"],
            "bat_exit_velo_diff":         team_h["bat_exit_velo"] - team_a["bat_exit_velo"],
            "xwoba_diff":                 team_h["xwoba"]         - team_a["xwoba"],
            "home_starter_velo":           sp_h["velo"],
            "away_starter_velo":           sp_a["velo"],
            "home_starter_k_pct":          sp_h["k_pct"],
            "away_starter_k_pct":          sp_a["k_pct"],
            "home_starter_bb_pct":         sp_h["bb_pct"],
            "away_starter_bb_pct":         sp_a["bb_pct"],
            "home_starter_k_minus_bb_pct": sp_h["k_minus_bb_pct"],
            "away_starter_k_minus_bb_pct": sp_a["k_minus_bb_pct"],
            "home_starter_xwoba_against":  sp_h["xwoba_against"],
            "away_starter_xwoba_against":  sp_a["xwoba_against"],
            "home_starter_ip":             sp_h["ip"],
            "away_starter_ip":             sp_a["ip"],
            "starter_velo_diff":           sp_h["velo"]           - sp_a["velo"],
            "starter_k_pct_diff":          sp_h["k_pct"]          - sp_a["k_pct"],
            "starter_bb_pct_diff":         sp_h["bb_pct"]         - sp_a["bb_pct"],
            "starter_k_minus_bb_pct_diff": sp_h["k_minus_bb_pct"] - sp_a["k_minus_bb_pct"],
            "starter_xwoba_diff":          sp_h["xwoba_against"]  - sp_a["xwoba_against"],
            "starter_ip_diff":             sp_h["ip"]             - sp_a["ip"],
            "home_elo_prob":              elo_prob,
            "elo_diff":                   elo_h - elo_a,
            "home_rest_days":             rest_h,
            "away_rest_days":             rest_a,
            "rest_days_diff":             rest_h - rest_a,
            "home_park_factor":           PARK_FACTORS.get(home, 1.0),
            "home_bullpen_k_pct":         bp_h,
            "away_bullpen_k_pct":         bp_a,
            "bullpen_k_pct_diff":         bp_h - bp_a,
            "home_win_pct_l15":           rec_h["win_pct"],
            "away_win_pct_l15":           rec_a["win_pct"],
            "home_run_diff_l15":          rec_h["run_diff"],
            "away_run_diff_l15":          rec_a["run_diff"],
            "win_pct_diff":               rec_h["win_pct"]  - rec_a["win_pct"],
            "run_diff_diff":              rec_h["run_diff"] - rec_a["run_diff"],
            "home_rs_l15":                rec_h["runs_scored"],
            "away_rs_l15":                rec_a["runs_scored"],
            "home_ra_l15":                rec_h["runs_allowed"],
            "away_ra_l15":                rec_a["runs_allowed"],
            "rs_diff":                    rec_h["runs_scored"]  - rec_a["runs_scored"],
            "ra_diff":                    rec_h["runs_allowed"] - rec_a["runs_allowed"],
            "home_starter_fip":           sp_h["fip"],
            "away_starter_fip":           sp_a["fip"],
            "starter_fip_diff":           sp_h["fip"] - sp_a["fip"],
            "home_opp_xwoba_l5":          opp_xwoba_h,
            "away_opp_xwoba_l5":          opp_xwoba_a,
            "opp_xwoba_diff":             opp_xwoba_h - opp_xwoba_a,
            "wind_speed_mph":             wx["wind_speed_mph"],
            "wind_component_out":         wx["wind_component_out"],
            "temperature_f":              wx["temperature_f"],
            "precip_probability":         wx["precip_probability"],
        })

    if not rows:
        print("Could not build features for any of today's games.")
        return []

    df = pd.DataFrame(rows).dropna(subset=FEATURES)
    if df.empty:
        print("All games dropped after NaN filter.")
        return []

    # Blended prediction: 55% Platt-calibrated XGBoost + 45% Elo
    xgb_prob = predict_prob(df[FEATURES])
    elo_prob  = df["home_elo_prob"].values
    blended   = np.clip(0.55 * xgb_prob + 0.45 * elo_prob, 0.25, 0.82)
    df["model_prob_home"] = blended
    df["model_prob_away"] = 1 - df["model_prob_home"]

    predictions: list[dict] = []
    ev_bets:     list[dict] = []

    for _, row in df.iterrows():
        home   = row["home_team"]
        away   = row["away_team"]
        prob_h = float(row["model_prob_home"])
        prob_a = float(row["model_prob_away"])
        elo_p  = float(row["home_elo_prob"])
        odict  = odds_lookup.get((home, away))

        home_odds_display = round(odict["home_odds"], 3) if odict else None
        away_odds_display = round(odict["away_odds"], 3) if odict else None
        ev_h = ev_a = None
        mkt_home_prob = consensus_prob = None
        pinnacle_home_prob = None

        if odict:
            # ── Market probability baseline: Pinnacle → consensus → DraftKings ──
            pin_h = odict.get("pinnacle_home_odds")
            pin_a = odict.get("pinnacle_away_odds")
            if pin_h and pin_a:
                pinnacle_home_prob = _vig_free_prob(pin_h, pin_a)
                mkt_home_prob      = pinnacle_home_prob
            elif odict.get("consensus_home_prob"):
                mkt_home_prob = odict["consensus_home_prob"]
            else:
                mkt_home_prob = _vig_free_prob(odict["home_odds"], odict["away_odds"])

            mkt_away_prob = 1.0 - mkt_home_prob
            consensus_prob = odict.get("consensus_home_prob") or mkt_home_prob

            # ── EV against best available odds ────────────────────────────────
            best_h = odict.get("best_home_odds") or odict["home_odds"]
            best_a = odict.get("best_away_odds") or odict["away_odds"]
            best_h_book = odict.get("best_home_book") or "draftkings"
            best_a_book = odict.get("best_away_book") or "draftkings"

            ev_h = round((prob_h * (best_h - 1) * STAKE) - (prob_a * STAKE), 2)
            ev_a = round((prob_a * (best_a - 1) * STAKE) - (prob_h * STAKE), 2)

            # ── Line movement direction ───────────────────────────────────────
            # Positive = current Pinnacle prob moved TOWARD home vs opening
            opening_pin_h = odict.get("opening_pinnacle_home_prob")
            line_move_home = None
            if opening_pin_h and pinnacle_home_prob:
                line_move_home = pinnacle_home_prob - opening_pin_h  # +pp = toward home

            for side, prob, dec_odds, ev, mkt_prob, best_odds, best_book, lm_sign in [
                ("home", prob_h, best_h, ev_h, mkt_home_prob, best_h, best_h_book, +1),
                ("away", prob_a, best_a, ev_a, mkt_away_prob, best_a, best_a_book, -1),
            ]:
                edge = prob - mkt_prob
                if not (ev >= EV_THRESHOLD and MIN_MARKET_EDGE <= edge <= MAX_MARKET_EDGE):
                    continue

                # Line movement filter: skip if closing line moved ≥3pp against our side
                if line_move_home is not None:
                    move_against = line_move_home * lm_sign * -1  # positive = against our bet
                    if move_against >= LINE_MOVE_VETO_PP:
                        print(f"  VETO {side} {home if side=='home' else away}: "
                              f"line moved {move_against*100:.1f}pp against.")
                        continue

                # Compute line_move_direction for display: +1 confirming, -1 against, 0 neutral
                lm_dir = 0
                if line_move_home is not None:
                    delta = line_move_home * lm_sign  # positive = market moved our way
                    if abs(delta) >= 0.015:
                        lm_dir = 1 if delta > 0 else -1

                ev_bets.append({
                    "date":             today,
                    "matchup":          f"{away} @ {home}",
                    "side":             side,
                    "team":             home if side == "home" else away,
                    "model_prob":       round(prob, 4),
                    "market_prob":      round(mkt_prob, 4),
                    "pinnacle_prob":    round(pinnacle_home_prob if side == "home" else (1 - pinnacle_home_prob), 4) if pinnacle_home_prob else None,
                    "edge_vs_market":   round(edge, 4),
                    "entry_odds":       round(dec_odds, 3),
                    "entry_book":       best_book,
                    "ev":               round(ev, 2),
                    "kelly_pct":        round(_kelly_pct(prob, dec_odds), 2),
                    "line_move_direction": lm_dir,
                    "result":           "TBD",
                })

        confidence = max(prob_h, prob_a)
        predictions.append({
            "date":                    today,
            "matchup":                 f"{away} @ {home}",
            "home_team":               home,
            "away_team":               away,
            "home_model_prob":         round(prob_h, 4),
            "away_model_prob":         round(prob_a, 4),
            "confidence":              round(confidence, 4),
            "home_elo_prob":           round(elo_p, 4),
            "home_odds":               home_odds_display,
            "away_odds":               away_odds_display,
            "market_implied_home_prob": round(mkt_home_prob, 4) if mkt_home_prob else None,
            "consensus_home_prob":     round(consensus_prob, 4) if consensus_prob else None,
            "best_home_odds":          round(odict["best_home_odds"], 3) if odict and odict.get("best_home_odds") else None,
            "best_away_odds":          round(odict["best_away_odds"], 3) if odict and odict.get("best_away_odds") else None,
            "best_home_book":          odict.get("best_home_book") if odict else None,
            "best_away_book":          odict.get("best_away_book") if odict else None,
            "ev_home":                 ev_h,
            "ev_away":                 ev_a,
            "result":                  "TBD",
        })

    predictions.sort(key=lambda x: x["confidence"], reverse=True)
    ev_bets.sort(key=lambda x: x["ev"], reverse=True)

    # ── Write predictions to PostgreSQL ──────────────────────────────────────
    try:
        from update_results import _ensure_table
        _ensure_table(engine)
        with engine.begin() as conn:
            for p in predictions:
                conn.execute(text("""
                    INSERT INTO mlb_predictions
                        (game_date, matchup, home_team, away_team,
                         home_model_prob, away_model_prob, confidence, home_elo_prob,
                         home_odds, away_odds, market_implied_home_prob, ev_home, ev_away)
                    VALUES
                        (:game_date, :matchup, :home_team, :away_team,
                         :home_model_prob, :away_model_prob, :confidence, :home_elo_prob,
                         :home_odds, :away_odds, :market_implied_home_prob, :ev_home, :ev_away)
                    ON CONFLICT (game_date, home_team, away_team) DO UPDATE SET
                        home_model_prob          = EXCLUDED.home_model_prob,
                        away_model_prob          = EXCLUDED.away_model_prob,
                        confidence               = EXCLUDED.confidence,
                        home_elo_prob            = EXCLUDED.home_elo_prob,
                        home_odds                = EXCLUDED.home_odds,
                        away_odds                = EXCLUDED.away_odds,
                        market_implied_home_prob = EXCLUDED.market_implied_home_prob,
                        ev_home                  = EXCLUDED.ev_home,
                        ev_away                  = EXCLUDED.ev_away,
                        updated_at               = NOW()
                """), {
                    "game_date": p["date"], "matchup": p["matchup"],
                    "home_team": p["home_team"], "away_team": p["away_team"],
                    "home_model_prob": p["home_model_prob"], "away_model_prob": p["away_model_prob"],
                    "confidence": p["confidence"], "home_elo_prob": p["home_elo_prob"],
                    "home_odds": p.get("home_odds"), "away_odds": p.get("away_odds"),
                    "market_implied_home_prob": p.get("market_implied_home_prob"),
                    "ev_home": p.get("ev_home"), "ev_away": p.get("ev_away"),
                })
        print(f"  {len(predictions)} prediction(s) written to mlb_predictions.")
    except Exception as e:
        print(f"  Warning: could not write predictions ({e})")

    # ── Write EV bets to mlb_ev_bets table ───────────────────────────────────
    if ev_bets:
        try:
            _ensure_ev_bets_table(engine)
            with engine.begin() as conn:
                for b in ev_bets:
                    conn.execute(text("""
                        INSERT INTO mlb_ev_bets
                            (game_date, matchup, side, team,
                             model_prob, market_prob, pinnacle_prob, edge_vs_market,
                             entry_odds, entry_book, ev, kelly_pct,
                             line_move_direction, result)
                        VALUES
                            (:game_date, :matchup, :side, :team,
                             :model_prob, :market_prob, :pinnacle_prob, :edge_vs_market,
                             :entry_odds, :entry_book, :ev, :kelly_pct,
                             :line_move_direction, 'TBD')
                        ON CONFLICT (game_date, matchup, side) DO UPDATE SET
                            model_prob           = EXCLUDED.model_prob,
                            market_prob          = EXCLUDED.market_prob,
                            pinnacle_prob        = EXCLUDED.pinnacle_prob,
                            edge_vs_market       = EXCLUDED.edge_vs_market,
                            entry_odds           = EXCLUDED.entry_odds,
                            entry_book           = EXCLUDED.entry_book,
                            ev                   = EXCLUDED.ev,
                            kelly_pct            = EXCLUDED.kelly_pct,
                            line_move_direction  = EXCLUDED.line_move_direction,
                            updated_at           = datetime('now')
                    """), {
                        "game_date":           b["date"],
                        "matchup":             b["matchup"],
                        "side":                b["side"],
                        "team":                b["team"],
                        "model_prob":          b["model_prob"],
                        "market_prob":         b["market_prob"],
                        "pinnacle_prob":       b.get("pinnacle_prob"),
                        "edge_vs_market":      b["edge_vs_market"],
                        "entry_odds":          b["entry_odds"],
                        "entry_book":          b.get("entry_book"),
                        "ev":                  b["ev"],
                        "kelly_pct":           b["kelly_pct"],
                        "line_move_direction": b.get("line_move_direction", 0),
                    })
            print(f"  {len(ev_bets)} EV bet(s) written to mlb_ev_bets.")
        except Exception as e:
            print(f"  Warning: could not write EV bets ({e})")

    # ── CSV backup ────────────────────────────────────────────────────────────
    pred_fieldnames = [
        "date", "matchup", "home_team", "away_team",
        "home_model_prob", "away_model_prob", "confidence", "home_elo_prob",
        "home_odds", "away_odds", "market_implied_home_prob",
        "ev_home", "ev_away", "result",
    ]
    existing: list[dict] = []
    if os.path.exists(PREDICTIONS_PATH):
        with open(PREDICTIONS_PATH, newline="") as f:
            existing = [r for r in csv.DictReader(f) if r.get("date") != today]
    with open(PREDICTIONS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pred_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing)
        writer.writerows(predictions)

    ev_fieldnames = [
        "date", "matchup", "side", "team",
        "model_prob", "market_prob", "edge_vs_market",
        "entry_odds", "entry_book", "ev", "kelly_pct", "line_move_direction", "result",
    ]
    ev_existing: list[dict] = []
    if os.path.exists(EV_BETS_PATH):
        with open(EV_BETS_PATH, newline="") as f:
            ev_existing = [r for r in csv.DictReader(f) if r.get("date") != today]
    with open(EV_BETS_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ev_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ev_existing)
        writer.writerows(ev_bets)

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  MLB PREDICTIONS  —  {today}  (sorted by confidence)")
    print(f"{'='*80}\n")
    print(f"  {'Matchup':<28} {'Conf':>6}  {'Pick':<22} {'Home%':>6} {'Away%':>6}  Elo")
    print(f"  {'-'*74}")
    for p in predictions:
        elo_mark = "✓" if abs(p["home_model_prob"] - p["home_elo_prob"]) < 0.08 else "~"
        pick = p["home_team"] if p["home_model_prob"] >= p["away_model_prob"] else p["away_team"]
        print(
            f"  {p['matchup']:<28} {p['confidence']*100:5.1f}%  "
            f"{pick:<22} {p['home_model_prob']*100:5.1f}% "
            f"{p['away_model_prob']*100:5.1f}%  "
            f"{elo_mark} {p['home_elo_prob']*100:.1f}%"
        )

    if ev_bets:
        print(f"\n{'='*80}")
        print(f"  +EV BETS  (EV>${EV_THRESHOLD:.0f}/100  |  edge {MIN_MARKET_EDGE*100:.0f}–{MAX_MARKET_EDGE*100:.0f}pp above Pinnacle)")
        print(f"{'='*80}\n")
        print(f"  {'Team':<22} {'Side':<5} {'Book':<12} {'Odds':>6}  {'Model%':>6}  {'Mkt%':>6}  {'Edge':>7}  {'EV/100':>8}  {'Kelly%':>6}  LM")
        print(f"  {'-'*86}")
        lm_sym = {1: "↑", -1: "↓", 0: "—"}
        for b in ev_bets:
            print(
                f"  {b['team']:<22} {b['side']:<5} "
                f"{b.get('entry_book','?'):<12} "
                f"{b['entry_odds']:>6.3f}  "
                f"{b['model_prob']*100:5.1f}%  "
                f"{b['market_prob']*100:5.1f}%  "
                f"{b['edge_vs_market']*100:+5.1f}pp  "
                f"${b['ev']:>7.2f}  "
                f"{b['kelly_pct']:>5.2f}%  "
                f"{lm_sym.get(b.get('line_move_direction', 0), '—')}"
            )
        print(f"\n  {len(ev_bets)} bet(s) logged.")
    else:
        print(f"\n  No +EV bets today (EV>${EV_THRESHOLD:.0f}/100 with "
              f"{MIN_MARKET_EDGE*100:.0f}–{MAX_MARKET_EDGE*100:.0f}pp Pinnacle edge).")

    return predictions


if __name__ == "__main__":
    run_predictions()
