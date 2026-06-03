TEAM_NAME_MAP: dict[str, str] = {
    "Atlanta Hawks":           "ATL",
    "Boston Celtics":          "BOS",
    "Brooklyn Nets":           "BKN",
    "Charlotte Hornets":       "CHA",
    "Chicago Bulls":           "CHI",
    "Cleveland Cavaliers":     "CLE",
    "Dallas Mavericks":        "DAL",
    "Denver Nuggets":          "DEN",
    "Detroit Pistons":         "DET",
    "Golden State Warriors":   "GSW",
    "Houston Rockets":         "HOU",
    "Indiana Pacers":          "IND",
    "Los Angeles Clippers":    "LAC",
    "Los Angeles Lakers":      "LAL",
    "Memphis Grizzlies":       "MEM",
    "Miami Heat":              "MIA",
    "Milwaukee Bucks":         "MIL",
    "Minnesota Timberwolves":  "MIN",
    "New Orleans Pelicans":    "NOP",
    "New York Knicks":         "NYK",
    "Oklahoma City Thunder":   "OKC",
    "Orlando Magic":           "ORL",
    "Philadelphia 76ers":      "PHI",
    "Phoenix Suns":            "PHX",
    "Portland Trail Blazers":  "POR",
    "Sacramento Kings":        "SAC",
    "San Antonio Spurs":       "SAS",
    "Toronto Raptors":         "TOR",
    "Utah Jazz":               "UTA",
    "Washington Wizards":      "WAS",
}

FEATURES: list[str] = [
    # Four Factors — home
    "home_efg", "home_tov_pct", "home_orb_pct", "home_ftr",
    # Four Factors — away
    "away_efg", "away_tov_pct", "away_orb_pct", "away_ftr",
    # Pace-adjusted ratings — home
    "home_ortg", "home_drtg", "home_net_rtg",
    # Pace-adjusted ratings — away
    "away_ortg", "away_drtg", "away_net_rtg",
    # Pace
    "home_pace", "away_pace", "pace_avg",
    # Differentials
    "efg_diff", "tov_pct_diff", "orb_pct_diff", "ftr_diff", "net_rtg_diff",
    # Rest
    "home_rest_days", "away_rest_days", "rest_diff",
    # Elo
    "home_elo", "away_elo", "elo_diff",
]

ROLLING_WINDOW = 10   # games used for rolling averages
MIN_GAMES      = 5    # minimum games required to generate a feature row
ELO_K          = 20   # Elo update magnitude
ELO_DEFAULT    = 1500.0
