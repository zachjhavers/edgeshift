"""
Shared feature engineering for mlb-engine.

Imported by both model_builder.py (training) and ev_engine.py (inference)
so the two paths stay in sync by construction.

Column contract for df_games:
  home_team, away_team, game_date, game_pk, home_win,
  final_home_score, final_away_score,
  home_pitch_velo, away_pitch_velo,
  home_bat_exit_velo, away_bat_exit_velo,
  home_xwoba, away_xwoba
"""

import pandas as pd

TEAM_WINDOW    = 15
STARTER_WINDOW = 5
BULLPEN_WINDOW = 10
ELO_K          = 20
ELO_HOME_ADV   = 35
ELO_INIT       = 1500
ELO_REGRESS    = 0.75

# Exponential decay factor for rolling stats.
# alpha=0.85: ~70% weight on last 5 games, ~90% on last 10 — strong recency bias.
EWMA_ALPHA = 0.85


def _ewma(series: pd.Series) -> float:
    """Last value of an exponentially weighted rolling mean (adjust=False = recursive)."""
    return float(series.ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1])


def build_elo_lookup(df: pd.DataFrame) -> dict:
    """Training path: {game_pk: {'home_elo_prob': float, 'elo_diff': float}}.
    Computed as-of-game (no lookahead).
    """
    elo, lookup, prev_year = {}, {}, None
    for _, game in df.sort_values("game_date").iterrows():
        year = game["game_date"].year
        if prev_year is not None and year != prev_year:
            for team in list(elo.keys()):
                elo[team] = elo[team] * ELO_REGRESS + ELO_INIT * (1 - ELO_REGRESS)
        prev_year = year

        home, away = game["home_team"], game["away_team"]
        elo_h = elo.get(home, ELO_INIT)
        elo_a = elo.get(away, ELO_INIT)
        e_home = 1 / (1 + 10 ** ((elo_a - elo_h - ELO_HOME_ADV) / 400))
        lookup[game["game_pk"]] = {"home_elo_prob": e_home, "elo_diff": elo_h - elo_a}

        actual = game["home_win"]
        elo[home] = elo_h + ELO_K * (actual - e_home)
        elo[away] = elo_a + ELO_K * ((1 - actual) - (1 - e_home))
    return lookup


def compute_current_elo(df: pd.DataFrame) -> dict:
    """Inference path: {team_name: current_elo_rating}.
    Call with all historical games up to (but not including) today.
    """
    elo, prev_year = {}, None
    for _, game in df.sort_values("game_date").iterrows():
        year = game["game_date"].year
        if prev_year is not None and year != prev_year:
            for team in list(elo.keys()):
                elo[team] = elo[team] * ELO_REGRESS + ELO_INIT * (1 - ELO_REGRESS)
        prev_year = year

        home, away = game["home_team"], game["away_team"]
        elo_h = elo.get(home, ELO_INIT)
        elo_a = elo.get(away, ELO_INIT)
        e_home = 1 / (1 + 10 ** ((elo_a - elo_h - ELO_HOME_ADV) / 400))

        actual = game["home_win"]
        elo[home] = elo_h + ELO_K * (actual - e_home)
        elo[away] = elo_a + ELO_K * ((1 - actual) - (1 - e_home))
    return elo


def get_team_rolling_stats(team: str, df_games: pd.DataFrame,
                           window: int = TEAM_WINDOW) -> dict | None:
    as_home = df_games[df_games["home_team"] == team][
        ["game_date", "home_pitch_velo", "home_bat_exit_velo", "home_xwoba"]
    ].rename(columns={
        "home_pitch_velo":    "pitch_velo",
        "home_bat_exit_velo": "bat_exit_velo",
        "home_xwoba":         "xwoba",
    })
    as_away = df_games[df_games["away_team"] == team][
        ["game_date", "away_pitch_velo", "away_bat_exit_velo", "away_xwoba"]
    ].rename(columns={
        "away_pitch_velo":    "pitch_velo",
        "away_bat_exit_velo": "bat_exit_velo",
        "away_xwoba":         "xwoba",
    })
    combined = pd.concat([as_home, as_away]).sort_values("game_date").tail(window)
    if len(combined) < 5:
        return None
    return {
        "pitch_velo":    _ewma(combined["pitch_velo"]),
        "bat_exit_velo": _ewma(combined["bat_exit_velo"]),
        "xwoba":         _ewma(combined["xwoba"]),
    }


def get_team_batting_advanced(team: str, df_games: pd.DataFrame,
                               window: int = TEAM_WINDOW) -> dict | None:
    """Rolling barrel rate and hard-hit rate for a team's offense."""
    as_home = df_games[df_games["home_team"] == team][
        ["game_date", "home_barrel_rate", "home_hard_hit_rate"]
    ].rename(columns={"home_barrel_rate": "barrel_rate", "home_hard_hit_rate": "hard_hit_rate"})
    as_away = df_games[df_games["away_team"] == team][
        ["game_date", "away_barrel_rate", "away_hard_hit_rate"]
    ].rename(columns={"away_barrel_rate": "barrel_rate", "away_hard_hit_rate": "hard_hit_rate"})
    combined = pd.concat([as_home, as_away]).dropna().sort_values("game_date").tail(window)
    if len(combined) < 5:
        return None
    return {
        "barrel_rate":   _ewma(combined["barrel_rate"]),
        "hard_hit_rate": _ewma(combined["hard_hit_rate"]),
    }


def get_team_record(team: str, df_games: pd.DataFrame,
                    window: int = TEAM_WINDOW) -> dict | None:
    home_games = df_games[df_games["home_team"] == team][
        ["game_date", "home_win", "final_home_score", "final_away_score"]
    ].copy()
    home_games["won"]          = home_games["home_win"].astype(float)
    home_games["run_diff"]     = home_games["final_home_score"] - home_games["final_away_score"]
    home_games["runs_scored"]  = home_games["final_home_score"]
    home_games["runs_allowed"] = home_games["final_away_score"]

    away_games = df_games[df_games["away_team"] == team][
        ["game_date", "home_win", "final_home_score", "final_away_score"]
    ].copy()
    away_games["won"]          = (1 - away_games["home_win"]).astype(float)
    away_games["run_diff"]     = away_games["final_away_score"] - away_games["final_home_score"]
    away_games["runs_scored"]  = away_games["final_away_score"]
    away_games["runs_allowed"] = away_games["final_home_score"]

    combined = pd.concat([
        home_games[["game_date", "won", "run_diff", "runs_scored", "runs_allowed"]],
        away_games[["game_date", "won", "run_diff", "runs_scored", "runs_allowed"]],
    ]).sort_values("game_date").tail(window)
    if len(combined) < 5:
        return None
    return {
        "win_pct":      _ewma(combined["won"]),
        "run_diff":     _ewma(combined["run_diff"]),
        "runs_scored":  _ewma(combined["runs_scored"]),
        "runs_allowed": _ewma(combined["runs_allowed"]),
    }


def get_opponent_xwoba(team: str, df_games: pd.DataFrame,
                       window: int = STARTER_WINDOW) -> float | None:
    """
    Rolling average xwOBA of the opponents this team's starters have recently faced.
    Captures opponent offensive quality — a starter's low xwOBA-against is more
    impressive against strong lineups than weak ones.

    For each of the last `window` starts for this team's home games:
      opponent = away team → their offensive xwOBA from prior games
    """
    home_games = df_games[df_games["home_team"] == team][
        ["game_date", "away_team", "away_xwoba"]
    ].sort_values("game_date").tail(window)
    away_games = df_games[df_games["away_team"] == team][
        ["game_date", "home_team", "home_xwoba"]
    ].rename(columns={"home_xwoba": "away_xwoba"}).sort_values("game_date").tail(window)

    combined = pd.concat([
        home_games[["game_date", "away_xwoba"]],
        away_games[["game_date", "away_xwoba"]],
    ]).sort_values("game_date").tail(window)

    if len(combined) < 2:
        return None
    return round(_ewma(combined["away_xwoba"]), 4)


_FIP_CONSTANT = 3.20


def get_starter_rolling_stats(pitcher_id, df_starters: pd.DataFrame,
                               game_date=None,
                               window: int = STARTER_WINDOW) -> dict | None:
    """
    game_date: pass pd.Timestamp during training to prevent lookahead.
               Pass None during inference (df_starters pre-filtered by SQL).
    Returns EWMA rolling stats over the last `window` starts including FIP.
    """
    if pitcher_id is None:
        return None
    prior = df_starters[df_starters["pitcher"] == pitcher_id]
    if game_date is not None:
        prior = prior[prior["game_date"] < game_date]
    prior = prior.sort_values("game_date").tail(window)
    if len(prior) < 2:
        return None

    k  = _ewma(prior["k_pct"])
    bb = _ewma(prior["bb_pct"]) if "bb_pct" in prior.columns else float("nan")

    ip_total  = prior["ip"].sum()       if "ip"        in prior.columns else 0.0
    hr_total  = prior["hr_count"].sum() if "hr_count"  in prior.columns else 0
    k_total   = prior["k_count"].sum()  if "k_count"   in prior.columns else 0
    bb_total  = prior["bb_count"].sum() if "bb_count"  in prior.columns else 0
    fb_total  = prior["fb_count"].sum() if "fb_count"  in prior.columns else 0
    hbp_total = prior["hbp_count"].sum() if "hbp_count" in prior.columns else 0
    gb_total  = prior["gb_count"].sum() if "gb_count"  in prior.columns else 0
    bip_total = prior["bip_count"].sum() if "bip_count" in prior.columns else 0

    fip = (
        (13 * hr_total + 3 * bb_total - 2 * k_total) / ip_total + _FIP_CONSTANT
        if ip_total > 0 else float("nan")
    )
    # xFIP: normalise HR using league-average HR/FB rate (~13%)
    _LG_HR_FB = 0.13
    xfip = (
        (13 * (fb_total * _LG_HR_FB) + 3 * (bb_total + hbp_total) - 2 * k_total) / ip_total + _FIP_CONSTANT
        if ip_total > 0 and fb_total >= 0 else float("nan")
    )
    gb_rate = (gb_total / bip_total) if bip_total > 0 else float("nan")

    return {
        "velo":            _ewma(prior["avg_velo"]),
        "k_pct":           k,
        "bb_pct":          bb,
        "k_minus_bb_pct":  k - bb,
        "xwoba_against":   _ewma(prior["xwoba_against"]),
        "ip":              _ewma(prior["ip"]) if "ip" in prior.columns else float("nan"),
        "fip":             fip,
        "xfip":            xfip,
        "gb_rate":         gb_rate,
    }


def get_rest_days(team: str, game_date, df_games: pd.DataFrame) -> float:
    prior = df_games[
        ((df_games["home_team"] == team) | (df_games["away_team"] == team)) &
        (df_games["game_date"] < game_date)
    ]
    if prior.empty:
        return 3.0
    return float(min((pd.Timestamp(game_date) - prior["game_date"].max()).days, 7))


def get_bullpen_k_pct(team: str, game_date, df_bullpen_agg: pd.DataFrame,
                      window: int = BULLPEN_WINDOW) -> float | None:
    prior = df_bullpen_agg[
        (df_bullpen_agg["team"] == team) &
        (df_bullpen_agg["game_date"] < game_date)
    ].tail(window)
    if len(prior) < 3:
        return None
    return float(_ewma(prior["bp_k_pct"]))


_LG_UMPIRE_K_RATE = 0.225  # league-average K-rate fallback


def build_umpire_k_lookup(df_umpire: pd.DataFrame) -> dict:
    """
    Build a lookup: {(game_date_str, home_team): umpire_k_rate}
    df_umpire must have columns: game_date, home_team, umpire_id, game_k_rate
    For each game, umpire_k_rate = EWMA of that umpire's prior game k-rates.
    """
    # Sort chronologically and compute rolling per-umpire k-rate (no lookahead)
    df_umpire = df_umpire.sort_values("game_date").reset_index(drop=True)
    lookup: dict[tuple, float] = {}
    ump_history: dict[int, list] = {}

    for _, row in df_umpire.iterrows():
        uid  = int(row["umpire_id"]) if pd.notna(row["umpire_id"]) else -1
        gk   = float(row["game_k_rate"]) if pd.notna(row["game_k_rate"]) else _LG_UMPIRE_K_RATE
        key  = (str(row["game_date"])[:10], row["home_team"])

        hist = ump_history.get(uid, [])
        if len(hist) >= 3:
            s = pd.Series(hist[-20:])
            rate = float(s.ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1])
        else:
            rate = _LG_UMPIRE_K_RATE

        lookup[key] = rate
        ump_history.setdefault(uid, []).append(gk)

    return lookup


def build_bullpen_agg(df_pitcher: pd.DataFrame) -> pd.DataFrame:
    """Aggregate non-starter K% by (team, game_date).

    Inning convention:
      Top of inning → HOME team is pitching.
      Bot of inning → AWAY team is pitching.
    """
    df_bp = df_pitcher[~df_pitcher["is_starter"]].copy()
    df_bp["team"] = df_bp.apply(
        lambda r: r["home_team"] if r["inning_topbot"] == "Top" else r["away_team"], axis=1
    )
    return (
        df_bp.groupby(["team", "game_date"])
        .agg(bp_k_pct=("k_pct", "mean"))
        .reset_index()
        .sort_values(["team", "game_date"])
    )
