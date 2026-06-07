"""
Constants and team name normalisation for the soccer engine.
The martj42 dataset uses full country names; The Odds API uses varying formats.
We normalise everything to a canonical short name.
"""

# How many recent matches to use for Dixon-Coles fitting
DC_WINDOW_MATCHES = 30       # per team, weighted by recency & importance
DC_BASE_GOALS     = 1.35     # average goals per team per match in international football

# Match importance weights (for weighted MLE)
IMPORTANCE = {
    "FIFA World Cup":              3.0,
    "UEFA Euro":                   2.5,
    "Copa América":                2.5,
    "African Cup of Nations":      2.0,
    "CONCACAF Gold Cup":           2.0,
    "AFC Asian Cup":               2.0,
    "FIFA World Cup qualification": 2.0,
    "UEFA Euro qualification":      1.5,
    "friendly":                    0.5,
    "Friendly":                    0.5,
}
DEFAULT_IMPORTANCE = 1.0

# Host nations get a neutral-venue home boost at the 2026 WC
HOST_NATIONS = {"United States", "Canada", "Mexico"}
HOST_ADVANTAGE = 0.12   # added to log-lambda for host side

# EV gates (same thresholds as MLB/NHL/NBA)
EV_THRESHOLD      = 5.0    # $/100
MIN_MARKET_EDGE   = 0.04
MAX_MARKET_EDGE   = 0.12   # soccer draws can be very uncertain — wider gate
MAX_KELLY         = 0.08   # cap at 8% of bankroll

# Normalise Odds API team names → martj42 / canonical names
ODDS_TO_CANONICAL = {
    # USA / CONCACAF
    "United States":          "United States",
    "USA":                    "United States",
    "US":                     "United States",
    "Mexico":                 "Mexico",
    "Canada":                 "Canada",
    "Costa Rica":             "Costa Rica",
    "Panama":                 "Panama",
    "Honduras":               "Honduras",
    "Jamaica":                "Jamaica",
    "El Salvador":            "El Salvador",
    "Trinidad and Tobago":    "Trinidad and Tobago",
    # South America
    "Brazil":                 "Brazil",
    "Argentina":              "Argentina",
    "Colombia":               "Colombia",
    "Uruguay":                "Uruguay",
    "Chile":                  "Chile",
    "Ecuador":                "Ecuador",
    "Peru":                   "Peru",
    "Venezuela":              "Venezuela",
    "Bolivia":                "Bolivia",
    "Paraguay":               "Paraguay",
    # Europe
    "England":                "England",
    "France":                 "France",
    "Spain":                  "Spain",
    "Germany":                "Germany",
    "Portugal":               "Portugal",
    "Netherlands":            "Netherlands",
    "Belgium":                "Belgium",
    "Italy":                  "Italy",
    "Switzerland":            "Switzerland",
    "Croatia":                "Croatia",
    "Denmark":                "Denmark",
    "Austria":                "Austria",
    "Poland":                 "Poland",
    "Serbia":                 "Serbia",
    "Czech Republic":         "Czech Republic",
    "Czechia":                "Czech Republic",
    "Slovakia":               "Slovakia",
    "Romania":                "Romania",
    "Hungary":                "Hungary",
    "Ukraine":                "Ukraine",
    "Scotland":               "Scotland",
    "Turkey":                 "Turkey",
    "Türkiye":                "Turkey",
    "Slovenia":               "Slovenia",
    "Albania":                "Albania",
    "Georgia":                "Georgia",
    "Wales":                  "Wales",
    "Northern Ireland":       "Northern Ireland",
    "Bosnia and Herzegovina": "Bosnia and Herzegovina",
    "Bosnia-Herzegovina":     "Bosnia and Herzegovina",
    "North Macedonia":        "North Macedonia",
    "Kosovo":                 "Kosovo",
    "Norway":                 "Norway",
    "Sweden":                 "Sweden",
    "Finland":                "Finland",
    # Africa
    "Morocco":                "Morocco",
    "Senegal":                "Senegal",
    "Nigeria":                "Nigeria",
    "Ghana":                  "Ghana",
    "Cameroon":               "Cameroon",
    "Tunisia":                "Tunisia",
    "Egypt":                  "Egypt",
    "Algeria":                "Algeria",
    "South Africa":           "South Africa",
    "Mali":                   "Mali",
    "Ivory Coast":            "Ivory Coast",
    "Côte d'Ivoire":          "Ivory Coast",
    "Cote d'Ivoire":          "Ivory Coast",
    "DR Congo":               "DR Congo",
    "Congo DR":               "DR Congo",
    "Cape Verde":             "Cape Verde",
    "Curaçao":                "Curaçao",
    "Curacao":                "Curaçao",
    "Tanzania":               "Tanzania",
    "Zambia":                 "Zambia",
    "Angola":                 "Angola",
    "Benin":                  "Benin",
    # Asia / Oceania
    "Japan":                  "Japan",
    "South Korea":            "South Korea",
    "Saudi Arabia":           "Saudi Arabia",
    "Iran":                   "Iran",
    "IR Iran":                "Iran",
    "Australia":              "Australia",
    "China":                  "China PR",
    "China PR":               "China PR",
    "Qatar":                  "Qatar",
    "Uzbekistan":             "Uzbekistan",
    "Indonesia":              "Indonesia",
    "Jordan":                 "Jordan",
    "Bahrain":                "Bahrain",
    "New Zealand":            "New Zealand",
    # CONCACAF (others)
    "Cuba":                   "Cuba",
    "Guatemala":              "Guatemala",
    "Haiti":                  "Haiti",
    # Middle East / other
    "Israel":                 "Israel",
    "Greece":                 "Greece",
    "Iraq":                   "Iraq",
}


def canonical(name: str) -> str:
    """Normalise a team name to its canonical form."""
    return ODDS_TO_CANONICAL.get(name, name)


def importance_weight(tournament: str) -> float:
    for key, w in IMPORTANCE.items():
        if key.lower() in tournament.lower():
            return w
    return DEFAULT_IMPORTANCE
