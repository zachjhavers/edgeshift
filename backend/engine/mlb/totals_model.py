"""
Train an XGBoost regressor to predict total runs scored (home + away).

At prediction time totals_ev_engine.py uses the stored model + residual std_dev
to compute P(over/under) for a given O/U line via a normal CDF approximation.

Run: python totals_model.py
Saves: xgb_mlb_totals.pkl  (dict with keys 'xgb', 'residual_std')
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from db import get_engine
from utils import TOTALS_FEATURES, PARK_FACTORS
from weather import LEAGUE_AVG_WEATHER
from feature_helpers import (
    build_elo_lookup,
    build_bullpen_agg,
    get_team_rolling_stats,
    get_team_record,
    get_starter_rolling_stats,
    get_bullpen_k_pct,
    get_rest_days,
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_mlb_totals.pkl")


def build_and_train_totals_model():
    print("--- MLB Totals Model Training ---")
    engine = get_engine()

    print("Loading game data...")
    df_games = pd.read_sql("""
        SELECT
            game_pk, game_date, home_team, away_team,
            MAX(home_score) AS final_home_score,
            MAX(away_score) AS final_away_score,
            AVG(CASE WHEN inning_topbot = 'Top' THEN release_speed END)                    AS home_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN release_speed END)                    AS away_pitch_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN launch_speed END)                     AS home_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Top' THEN launch_speed END)                     AS away_bat_exit_velo,
            AVG(CASE WHEN inning_topbot = 'Bot' THEN estimated_woba_using_speedangle END)  AS home_xwoba,
            AVG(CASE WHEN inning_topbot = 'Top' THEN estimated_woba_using_speedangle END)  AS away_xwoba
        FROM statcast_raw
        WHERE game_type = 'R'
        GROUP BY game_pk, game_date, home_team, away_team
        HAVING MAX(home_score) IS NOT NULL AND MAX(away_score) IS NOT NULL
        ORDER BY game_date
    """, engine, parse_dates=["game_date"])
    df_games["total_runs"] = df_games["final_home_score"] + df_games["final_away_score"]
    df_games = df_games.dropna(subset=["home_xwoba", "away_xwoba"]).reset_index(drop=True)
    print(f"  {len(df_games)} games loaded.")

    print("Loading weather cache...")
    weather_lookup: dict = {}
    try:
        df_wx = pd.read_sql(
            "SELECT game_date AS gd, home_team, wind_speed_mph, wind_component_out, "
            "temperature_f, precip_probability FROM weather_cache",
            engine,
        )
        for _, r in df_wx.iterrows():
            weather_lookup[(r["gd"], r["home_team"])] = {
                "wind_speed_mph":     float(r["wind_speed_mph"]),
                "wind_component_out": float(r["wind_component_out"]),
                "temperature_f":      float(r["temperature_f"]),
                "precip_probability": float(r["precip_probability"]),
            }
    except Exception as e:
        print(f"  Weather cache unavailable ({e}), using league averages.")

    print("Loading pitcher data...")
    df_pitcher = pd.read_sql("""
        SELECT
            game_pk, game_date, home_team, away_team, pitcher, inning_topbot,
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
        FROM statcast_raw WHERE game_type='R'
        GROUP BY game_pk, game_date, home_team, away_team, pitcher, inning_topbot
    """, engine, parse_dates=["game_date"])
    starter_idx    = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    starter_lookup = {(r["game_pk"], r["inning_topbot"]): r["pitcher"]
                      for _, r in df_starters.iterrows()}
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    print(f"Building features for {len(df_games)} games...")
    rows = []
    for _, game in df_games.iterrows():
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
        if not home_sp_id or not away_sp_id:
            continue
        sp_h = get_starter_rolling_stats(home_sp_id, df_starters, game_date=game["game_date"])
        sp_a = get_starter_rolling_stats(away_sp_id, df_starters, game_date=game["game_date"])
        if sp_h is None or sp_a is None:
            continue
        bp_h = get_bullpen_k_pct(game["home_team"], game["game_date"], df_bullpen_agg)
        bp_a = get_bullpen_k_pct(game["away_team"], game["game_date"], df_bullpen_agg)
        if bp_h is None or bp_a is None:
            continue

        rest_h = get_rest_days(game["home_team"], game["game_date"], df_games)
        rest_a = get_rest_days(game["away_team"], game["game_date"], df_games)
        date_str = str(game["game_date"].date())
        wx = weather_lookup.get((date_str, game["home_team"]), LEAGUE_AVG_WEATHER)

        rows.append({
            "total_runs":                 game["total_runs"],
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
            "home_park_factor":           PARK_FACTORS.get(game["home_team"], 1.0),
            "wind_component_out":         wx["wind_component_out"],
            "wind_speed_mph":             wx["wind_speed_mph"],
            "temperature_f":              wx["temperature_f"],
            "home_rest_days":             rest_h,
            "away_rest_days":             rest_a,
        })

    df = pd.DataFrame(rows).dropna(subset=TOTALS_FEATURES)
    print(f"Training examples after NaN filter: {len(df)}")

    n         = len(df)
    train_end = int(n * 0.80)
    X_train   = df.iloc[:train_end][TOTALS_FEATURES]
    y_train   = df.iloc[:train_end]["total_runs"]
    X_val     = df.iloc[train_end:][TOTALS_FEATURES]
    y_val     = df.iloc[train_end:]["total_runs"]
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    model = XGBRegressor(
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
        verbosity=0,
    )
    model.fit(X_train, y_train)

    preds        = model.predict(X_val)
    residuals    = y_val.values - preds
    residual_std = float(np.std(residuals))
    mae          = mean_absolute_error(y_val, preds)

    print(f"\nVal MAE:      {mae:.3f} runs")
    print(f"Residual std: {residual_std:.3f} runs  (used for P(over) calc)")
    print(f"Mean pred:    {preds.mean():.2f}  |  Mean actual: {y_val.mean():.2f}")

    print("\nTop 10 feature importances:")
    imp = pd.Series(model.feature_importances_, index=TOTALS_FEATURES).sort_values(ascending=False)
    for name, val in imp.head(10).items():
        print(f"  {name:<35} {val:.4f}")

    joblib.dump({"xgb": model, "residual_std": residual_std}, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    build_and_train_totals_model()
