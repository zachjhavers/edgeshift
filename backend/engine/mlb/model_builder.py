import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import joblib
from db import get_engine
from utils import FEATURES, PARK_FACTORS
from weather import LEAGUE_AVG_WEATHER

from feature_helpers import (
    build_elo_lookup,
    get_team_rolling_stats,
    get_team_record,
    get_starter_rolling_stats,
    get_rest_days,
    get_bullpen_k_pct,
    get_opponent_xwoba,
    build_bullpen_agg,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_mlb_v1.pkl")


def _expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE): probability-weighted mean absolute deviation
    between predicted confidence and observed accuracy across n_bins equal-width bins.
    Lower is better (0 = perfectly calibrated).
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        acc  = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def _tune_platt(model_raw, X_cal, y_cal, X_val, y_val) -> tuple:
    """
    Grid-search Platt scaling regularisation C on the calibration set,
    selecting the C that minimises Brier score on the validation set.
    Returns (best_platt, best_brier).
    """
    best_platt, best_brier = None, float("inf")
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
        raw_cal = model_raw.predict_proba(X_cal)[:, 1].reshape(-1, 1)
        platt   = LogisticRegression(C=C, max_iter=1000)
        platt.fit(raw_cal, y_cal)
        raw_val  = model_raw.predict_proba(X_val)[:, 1].reshape(-1, 1)
        proba    = platt.predict_proba(raw_val)[:, 1]
        brier    = brier_score_loss(y_val, proba)
        if brier < best_brier:
            best_brier, best_platt = brier, platt
    return best_platt, best_brier


def build_and_train_model():
    print("--- Initiating Model Training Pipeline ---")
    engine = get_engine()

    print("Pulling per-game Statcast data...")
    query = """
        SELECT
            game_pk,
            CAST(game_date AS DATE)  AS game_date,
            home_team, away_team,
            MAX(home_score)          AS final_home_score,
            MAX(away_score)          AS final_away_score,
            AVG(CASE WHEN inning_topbot = 'Top' THEN release_speed                    END) AS home_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN release_speed                    END) AS away_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN launch_speed                     END) AS home_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Top' THEN launch_speed                     END) AS away_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN estimated_woba_using_speedangle  END) AS home_xwoba,
            AVG(CASE WHEN inning_topbot = 'Top' THEN estimated_woba_using_speedangle  END) AS away_xwoba
        FROM statcast_raw
        WHERE game_type = 'R'
        GROUP BY game_pk, CAST(game_date AS DATE), home_team, away_team
        HAVING MAX(home_score) IS NOT NULL AND MAX(away_score) IS NOT NULL
        ORDER BY game_date
    """
    df_games = pd.read_sql(query, engine, parse_dates=["game_date"])
    df_games["home_win"] = (df_games["final_home_score"] > df_games["final_away_score"]).astype(int)
    base_cols = ["home_pitch_velo", "away_pitch_velo", "home_bat_exit_velo",
                 "away_bat_exit_velo", "home_xwoba", "away_xwoba"]
    df_games = df_games.dropna(subset=base_cols).reset_index(drop=True)

    # Load weather cache (populated by backfill_weather.py; fills with league avg if missing)
    print("Loading weather cache...")
    weather_lookup: dict[tuple, dict] = {}
    try:
        df_wx = pd.read_sql(
            "SELECT game_date::text AS gd, home_team, wind_speed_mph, wind_direction_deg, "
            "wind_component_out, temperature_f, precip_probability FROM weather_cache",
            engine,
        )
        for _, r in df_wx.iterrows():
            weather_lookup[(r["gd"], r["home_team"])] = {
                "wind_speed_mph":     float(r["wind_speed_mph"]),
                "wind_direction_deg": float(r["wind_direction_deg"]),
                "wind_component_out": float(r["wind_component_out"]),
                "temperature_f":      float(r["temperature_f"]),
                "precip_probability": float(r["precip_probability"]),
            }
        print(f"  {len(weather_lookup)} game-day weather entries loaded.")
    except Exception as e:
        print(f"  Warning: could not load weather_cache ({e}). Using league averages.")

    print("Computing Elo ratings...")
    elo_lookup = build_elo_lookup(df_games)

    print("Pulling pitcher stats...")
    pitcher_query = """
        SELECT
            game_pk,
            CAST(game_date AS DATE)  AS game_date,
            home_team, away_team,
            pitcher, inning_topbot,
            AVG(release_speed)       AS avg_velo,
            SUM(CASE WHEN events = 'strikeout' THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS k_pct,
            SUM(CASE WHEN events IN ('walk', 'intent_walk') THEN 1 ELSE 0 END)::float /
                NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS bb_pct,
            AVG(estimated_woba_using_speedangle) AS xwoba_against,
            SUM(CASE WHEN events = 'home_run'                              THEN 1 ELSE 0 END) AS hr_count,
            SUM(CASE WHEN events = 'strikeout'                             THEN 1 ELSE 0 END) AS k_count,
            SUM(CASE WHEN events IN ('walk', 'intent_walk')               THEN 1 ELSE 0 END) AS bb_count,
            COUNT(*)                 AS pitch_count,
            SUM(CASE
                WHEN events IN ('strikeout', 'field_out', 'force_out', 'sac_bunt', 'sac_fly',
                                'fielders_choice_out', 'other_out', 'caught_stealing_2b',
                                'caught_stealing_3b', 'caught_stealing_home',
                                'pickoff_caught_stealing_2b', 'pickoff_caught_stealing_3b',
                                'pickoff_caught_stealing_home', 'sac_bunt_double_play') THEN 1
                WHEN events IN ('grounded_into_double_play', 'strikeout_double_play',
                                'double_play', 'sac_fly_double_play') THEN 2
                WHEN events = 'triple_play' THEN 3
                ELSE 0 END) * 1.0 / 3.0 AS ip
        FROM statcast_raw
        WHERE game_type = 'R'
        GROUP BY game_pk, CAST(game_date AS DATE), home_team, away_team, pitcher, inning_topbot
    """
    df_pitcher  = pd.read_sql(pitcher_query, engine, parse_dates=["game_date"])
    starter_idx = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    starter_lookup = {(r["game_pk"], r["inning_topbot"]): r["pitcher"]
                      for _, r in df_starters.iterrows()}
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    print(f"Building rolling features for {len(df_games)} games...")
    rows = []
    for i, game in df_games.iterrows():
        prior = df_games[df_games["game_date"] < game["game_date"]]
        team_h = get_team_rolling_stats(game["home_team"], prior)
        team_a = get_team_rolling_stats(game["away_team"], prior)
        if team_h is None or team_a is None:
            continue

        rec_h = get_team_record(game["home_team"], prior)
        rec_a = get_team_record(game["away_team"], prior)
        if rec_h is None or rec_a is None:
            continue

        home_sp_id = starter_lookup.get((game["game_pk"], "Top"))
        away_sp_id = starter_lookup.get((game["game_pk"], "Bot"))
        if home_sp_id is None or away_sp_id is None:
            continue

        sp_h = get_starter_rolling_stats(home_sp_id, df_starters, game_date=game["game_date"])
        sp_a = get_starter_rolling_stats(away_sp_id, df_starters, game_date=game["game_date"])
        if sp_h is None or sp_a is None:
            continue

        bp_h = get_bullpen_k_pct(game["home_team"], game["game_date"], df_bullpen_agg)
        bp_a = get_bullpen_k_pct(game["away_team"], game["game_date"], df_bullpen_agg)
        if bp_h is None or bp_a is None:
            continue

        elo    = elo_lookup.get(game["game_pk"], {"home_elo_prob": 0.5, "elo_diff": 0.0})
        rest_h = get_rest_days(game["home_team"], game["game_date"], df_games)
        rest_a = get_rest_days(game["away_team"], game["game_date"], df_games)
        park_f = PARK_FACTORS.get(game["home_team"], 1.0)

        opp_xwoba_h = get_opponent_xwoba(game["home_team"], prior) or 0.320
        opp_xwoba_a = get_opponent_xwoba(game["away_team"], prior) or 0.320

        # Weather: use cache if available, otherwise league average
        date_str = str(game["game_date"].date())
        wx = weather_lookup.get((date_str, game["home_team"]), LEAGUE_AVG_WEATHER)

        rows.append({
            "home_win":                   game["home_win"],
            "home_pitch_velo":            team_h["pitch_velo"],
            "away_pitch_velo":            team_a["pitch_velo"],
            "home_bat_exit_velo":         team_h["bat_exit_velo"],
            "away_bat_exit_velo":         team_a["bat_exit_velo"],
            "home_xwoba":                 team_h["xwoba"],
            "away_xwoba":                 team_a["xwoba"],
            "pitch_velo_diff":            team_h["pitch_velo"]    - team_a["pitch_velo"],
            "bat_exit_velo_diff":         team_h["bat_exit_velo"] - team_a["bat_exit_velo"],
            "xwoba_diff":                 team_h["xwoba"]          - team_a["xwoba"],
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
            "starter_velo_diff":           sp_h["velo"]            - sp_a["velo"],
            "starter_k_pct_diff":          sp_h["k_pct"]           - sp_a["k_pct"],
            "starter_bb_pct_diff":         sp_h["bb_pct"]          - sp_a["bb_pct"],
            "starter_k_minus_bb_pct_diff": sp_h["k_minus_bb_pct"]  - sp_a["k_minus_bb_pct"],
            "starter_xwoba_diff":          sp_h["xwoba_against"]   - sp_a["xwoba_against"],
            "starter_ip_diff":             sp_h["ip"]              - sp_a["ip"],
            "home_elo_prob":              elo["home_elo_prob"],
            "elo_diff":                   elo["elo_diff"],
            "home_rest_days":             rest_h,
            "away_rest_days":             rest_a,
            "rest_days_diff":             rest_h - rest_a,
            "home_park_factor":           park_f,
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

    df = pd.DataFrame(rows)
    print(f"Training examples: {len(df)}")

    # 65/15/20 chronological split: train / calibrate / validate
    n         = len(df)
    train_end = int(n * 0.65)
    cal_end   = int(n * 0.80)

    X_train = df.iloc[:train_end][FEATURES]
    y_train = df.iloc[:train_end]["home_win"]
    X_cal   = df.iloc[train_end:cal_end][FEATURES]
    y_cal   = df.iloc[train_end:cal_end]["home_win"]
    X_val   = df.iloc[cal_end:][FEATURES]
    y_val   = df.iloc[cal_end:]["home_win"]

    print(f"Train: {len(X_train)}  |  Cal: {len(X_cal)}  |  Val: {len(X_val)}")

    model_raw = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=0.1,
        reg_lambda=1.5,
        reg_alpha=0.3,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    model_raw.fit(X_train, y_train)

    # Platt calibration: grid-search C on cal set, select by minimum Brier score on val set.
    # Published research (arxiv 2303.06021) shows calibration-optimised selection yields
    # 69.86% higher returns than accuracy-optimised selection.
    print("Tuning Platt calibration (minimising Brier score)...")
    platt, best_brier_during_tune = _tune_platt(model_raw, X_cal, y_cal, X_val, y_val)
    print(f"  Best Platt C: {platt.C:.4f}  |  Brier on val: {best_brier_during_tune:.4f}")

    def predict_prob(X):
        raw = model_raw.predict_proba(X)[:, 1].reshape(-1, 1)
        return platt.predict_proba(raw)[:, 1]

    proba    = predict_prob(X_val)
    accuracy = accuracy_score(y_val, (proba >= 0.5).astype(int))
    brier    = brier_score_loss(y_val, proba)
    ece      = _expected_calibration_error(np.array(y_val), proba)
    try:
        auc = roc_auc_score(y_val, proba)
    except Exception:
        auc = float("nan")

    print(f"\nAccuracy:       {accuracy * 100:.2f}%")
    print(f"Brier Score:    {brier:.4f}  (0.25 = random)  ← primary metric")
    print(f"ECE:            {ece:.4f}  (0.00 = perfect calibration)")
    print(f"ROC-AUC:        {auc:.4f}   (0.5 = random)")
    print(f"Mean predicted: {proba.mean():.3f}  (actual: {y_val.mean():.3f})")
    print(f"Pred std dev:   {proba.std():.4f}  (higher = more differentiation)")

    frac_pos, mean_pred = calibration_curve(y_val, proba, n_bins=5, strategy="quantile")
    print("\nCalibration check (predicted → actual win rate):")
    for pred, actual in zip(mean_pred, frac_pos):
        print(f"  {pred*100:4.1f}% predicted  →  {actual*100:4.1f}% actual")

    print("\nTop 10 feature importances:")
    imp = pd.Series(model_raw.feature_importances_, index=FEATURES).sort_values(ascending=False)
    for name, val in imp.head(10).items():
        print(f"  {name:<35} {val:.4f}")

    joblib.dump({"xgb": model_raw, "platt": platt}, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    build_and_train_model()
