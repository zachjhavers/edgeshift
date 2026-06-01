"""Team universe, feature definitions, and constants for the NHL engine."""

# 32 NHL teams (2024-25). Historical abbreviations included for backfill.
# ARI = Arizona Coyotes (relocated to UTA = Utah HC for 2024-25).
TEAMS: dict[str, dict] = {
    "ANA": {"name": "Anaheim Ducks",        "conference": "Western", "division": "Pacific"},
    "BOS": {"name": "Boston Bruins",         "conference": "Eastern", "division": "Atlantic"},
    "BUF": {"name": "Buffalo Sabres",        "conference": "Eastern", "division": "Atlantic"},
    "CGY": {"name": "Calgary Flames",        "conference": "Western", "division": "Pacific"},
    "CAR": {"name": "Carolina Hurricanes",   "conference": "Eastern", "division": "Metropolitan"},
    "CHI": {"name": "Chicago Blackhawks",    "conference": "Western", "division": "Central"},
    "COL": {"name": "Colorado Avalanche",    "conference": "Western", "division": "Central"},
    "CBJ": {"name": "Columbus Blue Jackets", "conference": "Eastern", "division": "Metropolitan"},
    "DAL": {"name": "Dallas Stars",          "conference": "Western", "division": "Central"},
    "DET": {"name": "Detroit Red Wings",     "conference": "Eastern", "division": "Atlantic"},
    "EDM": {"name": "Edmonton Oilers",       "conference": "Western", "division": "Pacific"},
    "FLA": {"name": "Florida Panthers",      "conference": "Eastern", "division": "Atlantic"},
    "LAK": {"name": "Los Angeles Kings",     "conference": "Western", "division": "Pacific"},
    "MIN": {"name": "Minnesota Wild",        "conference": "Western", "division": "Central"},
    "MTL": {"name": "Montreal Canadiens",    "conference": "Eastern", "division": "Atlantic"},
    "NSH": {"name": "Nashville Predators",   "conference": "Western", "division": "Central"},
    "NJD": {"name": "New Jersey Devils",     "conference": "Eastern", "division": "Metropolitan"},
    "NYI": {"name": "New York Islanders",    "conference": "Eastern", "division": "Metropolitan"},
    "NYR": {"name": "New York Rangers",      "conference": "Eastern", "division": "Metropolitan"},
    "OTT": {"name": "Ottawa Senators",       "conference": "Eastern", "division": "Atlantic"},
    "PHI": {"name": "Philadelphia Flyers",   "conference": "Eastern", "division": "Metropolitan"},
    "PIT": {"name": "Pittsburgh Penguins",   "conference": "Eastern", "division": "Metropolitan"},
    "SEA": {"name": "Seattle Kraken",        "conference": "Western", "division": "Pacific"},
    "SJS": {"name": "San Jose Sharks",       "conference": "Western", "division": "Pacific"},
    "STL": {"name": "St. Louis Blues",       "conference": "Western", "division": "Central"},
    "TBL": {"name": "Tampa Bay Lightning",   "conference": "Eastern", "division": "Atlantic"},
    "TOR": {"name": "Toronto Maple Leafs",   "conference": "Eastern", "division": "Atlantic"},
    "UTA": {"name": "Utah Hockey Club",      "conference": "Western", "division": "Pacific"},
    "VAN": {"name": "Vancouver Canucks",     "conference": "Western", "division": "Pacific"},
    "VGK": {"name": "Vegas Golden Knights",  "conference": "Western", "division": "Pacific"},
    "WSH": {"name": "Washington Capitals",   "conference": "Eastern", "division": "Metropolitan"},
    "WPG": {"name": "Winnipeg Jets",         "conference": "Western", "division": "Central"},
    # Historical — Arizona Coyotes (2020-21 → 2023-24)
    "ARI": {"name": "Arizona Coyotes",       "conference": "Western", "division": "Central"},
}

# All known team codes (current + historical)
ALL_TEAM_CODES = set(TEAMS.keys())

# Season date ranges for full historical backfill
SEASONS = [
    ("20202021", "2021-01-13", "2021-07-07"),   # COVID-shortened 56-game season
    ("20212022", "2021-10-12", "2022-07-11"),
    ("20222023", "2022-10-07", "2023-06-13"),
    ("20232024", "2023-10-10", "2024-06-24"),
    ("20242025", "2024-10-08", "2025-07-15"),
    ("20252026", "2025-10-07", "2026-07-15"),   # includes 2026 Stanley Cup playoffs
]

# Feature columns — must match the features table in db.py
FEATURES = [
    # Team rolling stats (last 10 games each side)
    "home_gf_10", "away_gf_10",
    "home_ga_10", "away_ga_10",
    "home_sf_10", "away_sf_10",
    "home_sa_10", "away_sa_10",
    "home_shot_pct_10", "away_shot_pct_10",
    "home_sv_pct_10",   "away_sv_pct_10",
    "home_pp_pct_10",   "away_pp_pct_10",
    "home_pk_pct_10",   "away_pk_pct_10",
    "home_win_pct_10",  "away_win_pct_10",
    # Differentials (home - away)
    "gf_diff", "sf_diff", "shot_pct_diff", "sv_pct_diff",
    "pp_pct_diff", "pk_pct_diff", "win_pct_diff",
    # Goalie rolling stats (last 5 starts for starting goalie)
    "home_goalie_sv_pct_5", "away_goalie_sv_pct_5",
    "home_goalie_gsaa_5",   "away_goalie_gsaa_5",
    "goalie_sv_pct_diff",   "goalie_gsaa_diff",
    # Rest & schedule
    "home_rest_days", "away_rest_days",
    "home_b2b",       "away_b2b",
    "rest_diff",
    # Elo
    "home_elo_prob", "elo_diff",
    # Calendar
    "is_playoff", "month",
    # PDO and shot share (new — retrain required)
    "home_pdo_10", "away_pdo_10", "pdo_diff",
    "home_shot_share_10", "away_shot_share_10", "shot_share_diff",
]

# Elo system constants — identical to MLB engine
ELO_K             = 20
ELO_HOME_ADV      = 35    # home advantage in Elo points
ELO_INIT          = 1500
ELO_SEASONAL_REG  = 0.75  # regression fraction toward mean each new season

# League-average save % used for GSAA computation
# Updated based on recent NHL averages (~2020-25)
LEAGUE_AVG_SV_PCT = 0.907

# Minimum games a team must have played before we compute features for them
MIN_TEAM_GAMES = 5
MIN_GOALIE_STARTS = 2

NHL_API_BASE = "https://api-web.nhle.com"
