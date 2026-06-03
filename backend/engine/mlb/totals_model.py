"""
MLB Half-Game Totals Model v2.

Architecture:
  One XGBoost regressor predicts runs scored by ONE team in ONE half of a game.
  Each game produces two training rows (home half, away half).
  Total prediction = predict(home_half) + predict(away_half).

Improvements over v1:
  - Half-game approach: offense features vs opposing pitcher (no confounding)
  - Barrel rate + hard-hit rate for offense quality
  - xFIP for starters (normalises HR/FB, more stable than FIP)
  - Ground ball rate for starters (suppresses extra-base hits)
  - Umpire K-rate tendency (0.3-0.5 run/game effect)
  - Negative binomial alpha fit (baseball is overdispersed vs Poisson/Normal)

Saves: xgb_mlb_totals.pkl  {'xgb', 'negbin_alpha', 'residual_std'}
"""

import os
import numpy as np
import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from scipy.stats import nbinom

from db import get_engine
from utils import HALF_GAME_FEATURES, PARK_FACTORS
from weather import LEAGUE_AVG_WEATHER
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "xgb_mlb_totals.pkl")

_GAME_SQL = """
    SELECT
        game_pk, game_date, home_team, away_team,
        MAX(home_score)  AS final_home_score,
        MAX(away_score)  AS final_away_score,
        AVG(CASE WHEN inning_topbot='Top' THEN release_speed END)                    AS home_pitch_velo,
        AVG(CASE WHEN inning_topbot='Bot' THEN release_speed END)                    AS away_pitch_velo,
        AVG(CASE WHEN inning_topbot='Bot' THEN launch_speed END)                     AS home_bat_exit_velo,
        AVG(CASE WHEN inning_topbot='Top' THEN launch_speed END)                     AS away_bat_exit_velo,
        AVG(CASE WHEN inning_topbot='Bot' THEN estimated_woba_using_speedangle END)  AS home_xwoba,
        AVG(CASE WHEN inning_topbot='Top' THEN estimated_woba_using_speedangle END)  AS away_xwoba,
        -- Barrel rate: exit velo >= 98, launch angle 26-30 (batted ball events only)
        SUM(CASE WHEN inning_topbot='Bot' AND launch_speed >= 98
                      AND launch_angle BETWEEN 26 AND 30 THEN 1.0 ELSE 0 END) /
            NULLIF(SUM(CASE WHEN inning_topbot='Bot' AND bb_type IS NOT NULL THEN 1.0 ELSE 0 END), 0)
            AS home_barrel_rate,
        SUM(CASE WHEN inning_topbot='Top' AND launch_speed >= 98
                      AND launch_angle BETWEEN 26 AND 30 THEN 1.0 ELSE 0 END) /
            NULLIF(SUM(CASE WHEN inning_topbot='Top' AND bb_type IS NOT NULL THEN 1.0 ELSE 0 END), 0)
            AS away_barrel_rate,
        -- Hard-hit rate: exit velo >= 95
        SUM(CASE WHEN inning_topbot='Bot' AND launch_speed >= 95 THEN 1.0 ELSE 0 END) /
            NULLIF(SUM(CASE WHEN inning_topbot='Bot' AND bb_type IS NOT NULL THEN 1.0 ELSE 0 END), 0)
            AS home_hard_hit_rate,
        SUM(CASE WHEN inning_topbot='Top' AND launch_speed >= 95 THEN 1.0 ELSE 0 END) /
            NULLIF(SUM(CASE WHEN inning_topbot='Top' AND bb_type IS NOT NULL THEN 1.0 ELSE 0 END), 0)
            AS away_hard_hit_rate
    FROM statcast_raw
    WHERE game_type = 'R'
    GROUP BY game_pk, game_date, home_team, away_team
    HAVING MAX(home_score) IS NOT NULL AND MAX(away_score) IS NOT NULL
    ORDER BY game_date
"""

_PITCHER_SQL = """
    SELECT
        game_pk, game_date, home_team, away_team, pitcher, inning_topbot,
        AVG(release_speed)  AS avg_velo,
        CAST(SUM(CASE WHEN events='strikeout'                THEN 1 ELSE 0 END) AS REAL) /
            NULLIF(SUM(CASE WHEN events IS NOT NULL          THEN 1 ELSE 0 END), 0) AS k_pct,
        CAST(SUM(CASE WHEN events IN ('walk','intent_walk')  THEN 1 ELSE 0 END) AS REAL) /
            NULLIF(SUM(CASE WHEN events IS NOT NULL          THEN 1 ELSE 0 END), 0) AS bb_pct,
        AVG(estimated_woba_using_speedangle) AS xwoba_against,
        SUM(CASE WHEN events='home_run'                      THEN 1 ELSE 0 END) AS hr_count,
        SUM(CASE WHEN events='strikeout'                     THEN 1 ELSE 0 END) AS k_count,
        SUM(CASE WHEN events IN ('walk','intent_walk')       THEN 1 ELSE 0 END) AS bb_count,
        SUM(CASE WHEN events='hit_by_pitch'                  THEN 1 ELSE 0 END) AS hbp_count,
        SUM(CASE WHEN bb_type='fly_ball'                     THEN 1 ELSE 0 END) AS fb_count,
        SUM(CASE WHEN bb_type='ground_ball'                  THEN 1 ELSE 0 END) AS gb_count,
        SUM(CASE WHEN bb_type IS NOT NULL                    THEN 1 ELSE 0 END) AS bip_count,
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
    GROUP BY game_pk, game_date, home_team, away_team, pitcher, inning_topbot
"""

_UMPIRE_SQL = """
    SELECT
        game_date,
        home_team,
        MAX(umpire) AS umpire_id,
        CAST(SUM(CASE WHEN events='strikeout' THEN 1 ELSE 0 END) AS REAL) /
            NULLIF(SUM(CASE WHEN events IS NOT NULL THEN 1 ELSE 0 END), 0) AS game_k_rate
    FROM statcast_raw
    WHERE game_type='R' AND umpire IS NOT NULL
    GROUP BY game_pk, game_date, home_team
"""


def _build_half_rows(df_games, df_starters, starter_lookup, df_bullpen_agg,
                     umpire_lookup, weather_lookup):
    """Build two training rows per game (home half, away half)."""
    rows = []
    for _, game in df_games.iterrows():
        prior = df_games[df_games["game_date"] < game["game_date"]]
        date_str = str(game["game_date"].date())

        # Park + weather (same for both halves)
        park_f = PARK_FACTORS.get(game["home_team"], 1.0)
        wx     = weather_lookup.get((date_str, game["home_team"]), LEAGUE_AVG_WEATHER)
        ump_k  = umpire_lookup.get((date_str, game["home_team"]), _LG_UMPIRE_K_RATE)

        # Starter IDs
        home_sp_id = starter_lookup.get((game["game_pk"], "Top"))  # home team pitches in Top
        away_sp_id = starter_lookup.get((game["game_pk"], "Bot"))  # away team pitches in Bot
        if not home_sp_id or not away_sp_id:
            continue

        # Pitching stats (home pitcher faces away batters; away pitcher faces home batters)
        home_sp = get_starter_rolling_stats(home_sp_id, df_starters, game_date=game["game_date"])
        away_sp = get_starter_rolling_stats(away_sp_id, df_starters, game_date=game["game_date"])
        if home_sp is None or away_sp is None:
            continue

        home_bp = get_bullpen_k_pct(game["home_team"], game["game_date"], df_bullpen_agg)
        away_bp = get_bullpen_k_pct(game["away_team"], game["game_date"], df_bullpen_agg)
        if home_bp is None or away_bp is None:
            continue

        # Batting advanced stats
        home_bat = get_team_batting_advanced(game["home_team"], prior)
        away_bat = get_team_batting_advanced(game["away_team"], prior)
        home_rec = get_team_record(game["home_team"], prior)
        away_rec = get_team_record(game["away_team"], prior)
        if not home_bat or not away_bat or not home_rec or not away_rec:
            continue

        # xwOBA for each team's offense
        home_off = get_team_rolling_stats(game["home_team"], prior)
        away_off = get_team_rolling_stats(game["away_team"], prior)
        if not home_off or not away_off:
            continue

        rest_h = get_rest_days(game["home_team"], game["game_date"], df_games)
        rest_a = get_rest_days(game["away_team"], game["game_date"], df_games)

        common = {
            "home_park_factor": park_f,
            "wind_component_out": wx["wind_component_out"],
            "temperature_f":      wx["temperature_f"],
            "umpire_k_rate":      ump_k,
        }

        # HOME HALF: home team bats, away starter pitches
        rows.append({
            **common,
            "runs_scored":             float(game["final_home_score"]),
            "off_xwoba":               home_off["xwoba"],
            "off_rs_l15":              home_rec["runs_scored"],
            "off_barrel_rate":         home_bat["barrel_rate"],
            "off_hard_hit_rate":       home_bat["hard_hit_rate"],
            "def_starter_k_pct":       away_sp["k_pct"],
            "def_starter_bb_pct":      away_sp["bb_pct"],
            "def_starter_xfip":        away_sp["xfip"],
            "def_starter_xwoba_against": away_sp["xwoba_against"],
            "def_starter_gb_rate":     away_sp["gb_rate"],
            "def_bullpen_k_pct":       away_bp,
            "is_home":                 1.0,
            "team_rest_days":          rest_h,
        })

        # AWAY HALF: away team bats, home starter pitches
        rows.append({
            **common,
            "runs_scored":             float(game["final_away_score"]),
            "off_xwoba":               away_off["xwoba"],
            "off_rs_l15":              away_rec["runs_scored"],
            "off_barrel_rate":         away_bat["barrel_rate"],
            "off_hard_hit_rate":       away_bat["hard_hit_rate"],
            "def_starter_k_pct":       home_sp["k_pct"],
            "def_starter_bb_pct":      home_sp["bb_pct"],
            "def_starter_xfip":        home_sp["xfip"],
            "def_starter_xwoba_against": home_sp["xwoba_against"],
            "def_starter_gb_rate":     home_sp["gb_rate"],
            "def_bullpen_k_pct":       home_bp,
            "is_home":                 0.0,
            "team_rest_days":          rest_a,
        })

    return rows


def build_and_train_totals_model():
    print("--- MLB Totals Model v2 (half-game) ---")
    engine = get_engine()

    print("Loading game data...")
    df_games = pd.read_sql(_GAME_SQL, engine, parse_dates=["game_date"])
    df_games["home_win"] = (df_games["final_home_score"] > df_games["final_away_score"]).astype(int)
    df_games = df_games.dropna(subset=["home_xwoba", "away_xwoba"]).reset_index(drop=True)
    print(f"  {len(df_games)} games loaded.")

    print("Loading weather cache...")
    weather_lookup: dict = {}
    try:
        df_wx = pd.read_sql(
            "SELECT game_date AS gd, home_team, wind_speed_mph, wind_component_out, "
            "temperature_f, precip_probability FROM weather_cache", engine,
        )
        for _, r in df_wx.iterrows():
            weather_lookup[(r["gd"], r["home_team"])] = {
                "wind_speed_mph":     float(r["wind_speed_mph"]),
                "wind_component_out": float(r["wind_component_out"]),
                "temperature_f":      float(r["temperature_f"]),
                "precip_probability": float(r["precip_probability"]),
            }
    except Exception:
        pass

    print("Loading umpire data...")
    umpire_lookup: dict = {}
    try:
        df_ump = pd.read_sql(_UMPIRE_SQL, engine, parse_dates=["game_date"])
        df_ump["game_date"] = df_ump["game_date"].astype(str).str[:10]
        umpire_lookup = build_umpire_k_lookup(df_ump)
        print(f"  {len(umpire_lookup)} umpire-game entries loaded.")
    except Exception as e:
        print(f"  Umpire data unavailable ({e}), using league averages.")

    print("Loading pitcher data...")
    df_pitcher = pd.read_sql(_PITCHER_SQL, engine, parse_dates=["game_date"])
    starter_idx    = df_pitcher.groupby(["game_pk", "inning_topbot"])["pitch_count"].idxmax()
    df_pitcher["is_starter"] = False
    df_pitcher.loc[starter_idx, "is_starter"] = True
    df_starters    = df_pitcher.loc[starter_idx].reset_index(drop=True)
    starter_lookup = {(r["game_pk"], r["inning_topbot"]): r["pitcher"]
                      for _, r in df_starters.iterrows()}
    df_bullpen_agg = build_bullpen_agg(df_pitcher)

    print(f"Building half-game features for {len(df_games)} games (~{len(df_games)*2} rows)...")
    rows = _build_half_rows(df_games, df_starters, starter_lookup,
                            df_bullpen_agg, umpire_lookup, weather_lookup)

    df = pd.DataFrame(rows).dropna(subset=HALF_GAME_FEATURES)
    print(f"Training rows after NaN filter: {len(df)}")

    n         = len(df)
    train_end = int(n * 0.80)
    X_train   = df.iloc[:train_end][HALF_GAME_FEATURES]
    y_train   = df.iloc[:train_end]["runs_scored"]
    X_val     = df.iloc[train_end:][HALF_GAME_FEATURES]
    y_val     = df.iloc[train_end:]["runs_scored"]
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    model = XGBRegressor(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=6,
        gamma=0.15,
        reg_lambda=2.0,
        reg_alpha=0.4,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_train, y_train)

    half_preds = model.predict(X_val)
    half_mae   = mean_absolute_error(y_val, half_preds)

    # Pair up halves to get full-game totals for accuracy metrics
    val_df = df.iloc[train_end:].copy()
    val_df["pred_runs"] = half_preds
    # Pair consecutive rows (home half, away half) for each game
    game_pairs = []
    idx_list = val_df.index.tolist()
    for i in range(0, len(idx_list) - 1, 2):
        r1 = val_df.loc[idx_list[i]]
        r2 = val_df.loc[idx_list[i + 1]]
        game_pairs.append({
            "actual_total": r1["runs_scored"] + r2["runs_scored"],
            "pred_total":   r1["pred_runs"]   + r2["pred_runs"],
        })
    gdf = pd.DataFrame(game_pairs)
    total_mae  = mean_absolute_error(gdf["actual_total"], gdf["pred_total"])
    total_res  = gdf["actual_total"].values - gdf["pred_total"].values
    total_std  = float(np.std(total_res))

    # Fit negative binomial dispersion parameter from training totals
    train_df = df.iloc[:train_end].copy()
    # Pair training halves
    train_pairs = []
    tidx = train_df.index.tolist()
    for i in range(0, len(tidx) - 1, 2):
        r1 = train_df.loc[tidx[i]]
        r2 = train_df.loc[tidx[i + 1]]
        train_pairs.append(r1["runs_scored"] + r2["runs_scored"])
    actual_totals = np.array(train_pairs)
    mu_tot  = float(actual_totals.mean())
    var_tot = float(actual_totals.var())
    # NegBin: var = mu + mu^2/alpha  →  alpha = mu^2 / (var - mu)
    negbin_alpha = mu_tot ** 2 / max(var_tot - mu_tot, 0.1)
    print(f"\nHalf-game MAE:   {half_mae:.3f} runs")
    print(f"Full-game MAE:   {total_mae:.3f} runs")
    print(f"Residual std:    {total_std:.3f} runs")
    print(f"NegBin alpha:    {negbin_alpha:.2f}  (higher = closer to Poisson)")
    print(f"Train totals — mean: {mu_tot:.2f}  std: {np.sqrt(var_tot):.2f}")

    # Direction accuracy on validation totals
    print("\nDirection accuracy (validation, using NegBin for P(over)):")
    for line in [7.5, 8.0, 8.5, 9.0, 9.5]:
        floor_l = int(line)
        r  = negbin_alpha
        correct = 0
        for _, row in gdf.iterrows():
            mu   = max(row["pred_total"], 0.5)
            p    = negbin_alpha / (negbin_alpha + mu)
            p_ov = float(nbinom.sf(floor_l, r, p))
            pred_over   = p_ov > 0.5
            actual_over = row["actual_total"] > line
            if pred_over == actual_over:
                correct += 1
        acc = correct / len(gdf)
        print(f"  Line {line}: {acc*100:.1f}%  (n={len(gdf)})")

    print("\nTop 10 feature importances:")
    imp = pd.Series(model.feature_importances_, index=HALF_GAME_FEATURES).sort_values(ascending=False)
    for name, val in imp.head(10).items():
        print(f"  {name:<35} {val:.4f}")

    joblib.dump({
        "xgb":          model,
        "negbin_alpha":  negbin_alpha,
        "residual_std":  total_std,
    }, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")


if __name__ == "__main__":
    build_and_train_totals_model()
