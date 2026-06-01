# utils.py

# Feature columns shared between model_builder.py, backtest.py, and ev_engine.py.
# Order matters — must be identical at training and inference time.
FEATURES = [
    # Team rolling stats (last 15 games, EWMA-weighted)
    "home_pitch_velo",
    "away_pitch_velo",
    "home_bat_exit_velo",
    "away_bat_exit_velo",
    "home_xwoba",
    "away_xwoba",
    "pitch_velo_diff",
    "bat_exit_velo_diff",
    "xwoba_diff",
    # Starting pitcher rolling stats (last 5 starts, EWMA-weighted)
    "home_starter_velo",
    "away_starter_velo",
    "home_starter_k_pct",
    "away_starter_k_pct",
    "home_starter_bb_pct",
    "away_starter_bb_pct",
    "home_starter_k_minus_bb_pct",
    "away_starter_k_minus_bb_pct",
    "home_starter_xwoba_against",
    "away_starter_xwoba_against",
    "home_starter_ip",
    "away_starter_ip",
    "starter_velo_diff",
    "starter_k_pct_diff",
    "starter_bb_pct_diff",
    "starter_k_minus_bb_pct_diff",
    "starter_xwoba_diff",
    "starter_ip_diff",
    # Elo pre-game win probability and rating gap
    "home_elo_prob",
    "elo_diff",
    # Rest days since last game (capped at 7)
    "home_rest_days",
    "away_rest_days",
    "rest_days_diff",
    # Ballpark run environment (Coors=1.28, Petco=0.91, league avg=1.0)
    "home_park_factor",
    # Bullpen K% rolling last 10 games (non-starters, EWMA-weighted)
    "home_bullpen_k_pct",
    "away_bullpen_k_pct",
    "bullpen_k_pct_diff",
    # Win%, run differential, and runs scored/allowed last 15 games (EWMA)
    "home_win_pct_l15",
    "away_win_pct_l15",
    "home_run_diff_l15",
    "away_run_diff_l15",
    "win_pct_diff",
    "run_diff_diff",
    "home_rs_l15",
    "away_rs_l15",
    "home_ra_l15",
    "away_ra_l15",
    "rs_diff",
    "ra_diff",
    # Starter FIP (Fielding Independent Pitching) — strips defense/luck from ERA
    "home_starter_fip",
    "away_starter_fip",
    "starter_fip_diff",
    # Opponent offensive quality — average xwOBA of offenses the starter has faced.
    # Adjusts xwoba_against for the strength of opponents: a low xwOBA-against
    # against weak lineups is less informative than the same mark vs. strong ones.
    "home_opp_xwoba_l5",
    "away_opp_xwoba_l5",
    "opp_xwoba_diff",
    # Weather features — from Open-Meteo forecast at game time.
    # wind_component_out: positive = blowing toward OF (hitter-friendly), negative = blowing in.
    # Roof/dome parks set all weather features to neutral constants.
    "wind_speed_mph",
    "wind_component_out",
    "temperature_f",
    "precip_probability",
]

# Park run factors — 2024-25 approximations. Higher = more offense.
PARK_FACTORS = {
    "COL": 1.28, "TEX": 1.12, "CIN": 1.10, "MIL": 1.06,
    "BOS": 1.06, "ATL": 1.04, "CHC": 1.03, "NYY": 1.02,
    "PHI": 1.01, "HOU": 1.00, "STL": 0.99, "CLE": 0.99,
    "DET": 0.99, "MIN": 0.99, "PIT": 0.98, "BAL": 0.98,
    "CWS": 0.98, "LAD": 0.97, "TOR": 0.97, "KC":  0.96,
    "WSH": 0.96, "NYM": 0.96, "LAA": 0.96, "ATH": 0.96,
    "TB":  0.95, "SEA": 0.95, "AZ":  0.95, "MIA": 0.94,
    "SF":  0.93, "SD":  0.91,
}

# Maps The Odds API full names → pybaseball/Baseball Savant abbreviations.
MLB_TEAM_MAP = {
    "Arizona Diamondbacks":  "AZ",
    "Atlanta Braves":        "ATL",
    "Baltimore Orioles":     "BAL",
    "Boston Red Sox":        "BOS",
    "Chicago Cubs":          "CHC",
    "Chicago White Sox":     "CWS",
    "Cincinnati Reds":       "CIN",
    "Cleveland Guardians":   "CLE",
    "Colorado Rockies":      "COL",
    "Detroit Tigers":        "DET",
    "Houston Astros":        "HOU",
    "Kansas City Royals":    "KC",
    "Los Angeles Angels":    "LAA",
    "Los Angeles Dodgers":   "LAD",
    "Miami Marlins":         "MIA",
    "Milwaukee Brewers":     "MIL",
    "Minnesota Twins":       "MIN",
    "New York Mets":         "NYM",
    "New York Yankees":      "NYY",
    "Oakland Athletics":     "ATH",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates":    "PIT",
    "San Diego Padres":      "SD",
    "San Francisco Giants":  "SF",
    "Seattle Mariners":      "SEA",
    "St. Louis Cardinals":   "STL",
    "Tampa Bay Rays":        "TB",
    "Texas Rangers":         "TEX",
    "Toronto Blue Jays":     "TOR",
    "Washington Nationals":  "WSH",
    "Athletics":             "ATH",
}

# Maps MLB Stats API abbreviations → Statcast abbreviations where they differ.
STATSAPI_TO_STATCAST = {
    "ARI": "AZ",
    "KCR": "KC",
    "OAK": "ATH",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "WSN": "WSH",
    "WAS": "WSH",
}
