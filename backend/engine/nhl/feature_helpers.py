"""
Shared feature engineering functions used by feature_builder.py and predict.py.
"""

import math
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from utils import (
    ELO_HOME_ADV, ELO_INIT, ELO_K, ELO_SEASONAL_REG,
    LEAGUE_AVG_SV_PCT, MIN_GOALIE_STARTS, MIN_TEAM_GAMES,
)

EWMA_ALPHA = 0.85   # ~70% weight on the last 5 games


def _ewma(series: pd.Series) -> float:
    """Exponentially weighted mean (alpha=0.85). Most recent observation gets ~70% weight vs 5 games ago."""
    if series.empty:
        return float("nan")
    return float(series.ewm(alpha=EWMA_ALPHA, adjust=False).mean().iloc[-1])


# ── Elo ───────────────────────────────────────────────────────────────────────

def _elo_win_prob(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400))


def build_elo_lookup(df_games: pd.DataFrame) -> dict[str, float]:
    """
    Build {game_id: home_elo_prob} from a time-sorted DataFrame of completed games.
    Each game gets the Elo state *before* that game.
    """
    ratings: dict[str, float] = {}
    last_season: Optional[str] = None
    lookup: dict[str, float] = {}

    for _, row in df_games.sort_values("game_date").iterrows():
        season = str(row["season"])
        home   = str(row["home_team"])
        away   = str(row["away_team"])
        result = str(row.get("result", "TBD"))

        if last_season is not None and season != last_season:
            for team in list(ratings):
                ratings[team] = ELO_INIT + ELO_SEASONAL_REG * (ratings[team] - ELO_INIT)
        last_season = season

        home_elo  = ratings.get(home, ELO_INIT)
        away_elo  = ratings.get(away, ELO_INIT)
        home_adj  = home_elo + ELO_HOME_ADV
        home_prob = _elo_win_prob(home_adj, away_elo)
        lookup[str(row["game_id"])] = home_prob

        if result == "HOME_WIN":
            actual = 1.0
        elif result == "AWAY_WIN":
            actual = 0.0
        else:
            continue

        delta = ELO_K * (actual - home_prob)
        ratings[home] = home_elo + delta
        ratings[away] = away_elo - delta

    return lookup


def compute_current_elo(df_games: pd.DataFrame) -> dict[str, float]:
    """Return current Elo rating for every team after processing all completed games."""
    ratings: dict[str, float] = {}
    last_season: Optional[str] = None

    for _, row in df_games.sort_values("game_date").iterrows():
        season = str(row["season"])
        home   = str(row["home_team"])
        away   = str(row["away_team"])
        result = str(row.get("result", "TBD"))

        if last_season is not None and season != last_season:
            for team in list(ratings):
                ratings[team] = ELO_INIT + ELO_SEASONAL_REG * (ratings[team] - ELO_INIT)
        last_season = season

        home_elo = ratings.get(home, ELO_INIT)
        away_elo = ratings.get(away, ELO_INIT)
        home_adj = home_elo + ELO_HOME_ADV

        if result == "HOME_WIN":
            actual = 1.0
        elif result == "AWAY_WIN":
            actual = 0.0
        else:
            continue

        home_prob = _elo_win_prob(home_adj, away_elo)
        delta = ELO_K * (actual - home_prob)
        ratings[home] = home_elo + delta
        ratings[away] = away_elo - delta

    return ratings


# ── Team rolling stats ────────────────────────────────────────────────────────

def get_team_rolling_stats(
    team: str,
    game_date: str,
    df_games: pd.DataFrame,
    df_stats: pd.DataFrame,
    window: int = 10,
) -> Optional[dict]:
    """
    Compute EWMA-weighted rolling team stats over the last `window` games.

    Returns dict with keys: gf, ga, sf, sa, shot_pct, sv_pct, pp_pct, pk_pct,
                            win_pct, pdo, shot_share.
    Returns None if fewer than MIN_TEAM_GAMES found.
    """
    team_games = df_games[
        ((df_games["home_team"] == team) | (df_games["away_team"] == team)) &
        (df_games["game_date"] < game_date) &
        (df_games["result"].isin(["HOME_WIN", "AWAY_WIN"]))
    ].sort_values("game_date").tail(window)

    if len(team_games) < MIN_TEAM_GAMES:
        return None

    game_ids = set(team_games["game_id"].astype(str))

    # Sort by game_date so EWMA weights the most recent games highest
    team_stats = df_stats[
        (df_stats["team_code"] == team) &
        (df_stats["game_id"].astype(str).isin(game_ids))
    ].merge(team_games[["game_id", "game_date"]], on="game_id", how="left").sort_values("game_date")

    if team_stats.empty:
        return None

    opp_stats = df_stats[
        (df_stats["team_code"] != team) &
        (df_stats["game_id"].astype(str).isin(game_ids))
    ].merge(team_games[["game_id", "game_date"]], on="game_id", how="left").sort_values("game_date")

    gf = _ewma(team_stats["goals"])
    sf = _ewma(team_stats["shots"])
    ga = _ewma(opp_stats["goals"]) if not opp_stats.empty else float("nan")
    sa = _ewma(opp_stats["shots"]) if not opp_stats.empty else float("nan")

    # Per-game PP% and PK% then EWMA
    pp_pct_series = team_stats.apply(
        lambda r: r["pp_goals"] / r["pp_opp"] if r["pp_opp"] > 0 else 0.0, axis=1
    )
    pk_pct_series = team_stats.apply(
        lambda r: 1.0 - r["pk_goals_against"] / r["pk_opp"] if r["pk_opp"] > 0 else 1.0, axis=1
    )
    pp_pct = _ewma(pp_pct_series)
    pk_pct = _ewma(pk_pct_series)

    shot_pct = gf / sf if sf > 0 else 0.0
    sv_pct   = 1 - (ga / sa) if sa > 0 and not math.isnan(ga) and not math.isnan(sa) else LEAGUE_AVG_SV_PCT

    # Win %
    wins = 0
    for _, g in team_games.iterrows():
        if g["result"] == "HOME_WIN" and g["home_team"] == team:
            wins += 1
        elif g["result"] == "AWAY_WIN" and g["away_team"] == team:
            wins += 1
    win_pct = wins / len(team_games)

    # PDO = shot% + sv% (luck indicator; regresses strongly toward 1.0)
    pdo = shot_pct + sv_pct

    # Shot share = sf / (sf + sa) — Corsi-equivalent possession proxy
    shot_share = sf / (sf + sa) if (sf + sa) > 0 else 0.5

    return {
        "gf":         round(gf, 3),
        "ga":         round(ga, 3),
        "sf":         round(sf, 3),
        "sa":         round(sa, 3),
        "shot_pct":   round(shot_pct, 4),
        "sv_pct":     round(sv_pct, 4),
        "pp_pct":     round(pp_pct, 4),
        "pk_pct":     round(pk_pct, 4),
        "win_pct":    round(win_pct, 4),
        "pdo":        round(pdo, 4),
        "shot_share": round(shot_share, 4),
    }


# ── Goalie rolling stats ──────────────────────────────────────────────────────

def get_starting_goalie_id(
    team: str,
    game_id: str,
    df_goalie_stats: pd.DataFrame,
) -> Optional[str]:
    rows = df_goalie_stats[
        (df_goalie_stats["game_id"].astype(str) == str(game_id)) &
        (df_goalie_stats["team_code"] == team)
    ]
    if rows.empty:
        return None
    starter = rows.loc[rows["shots_against"].idxmax()]
    return str(starter["goalie_id"])


def get_goalie_rolling_stats(
    goalie_id: str,
    game_date: str,
    df_games: pd.DataFrame,
    df_goalie_stats: pd.DataFrame,
    window: int = 5,
) -> Optional[dict]:
    """
    Compute EWMA-weighted sv% and GSAA for a goalie over their last `window` starts.

    Returns dict with keys: sv_pct, gsaa.
    """
    goalie_rows = df_goalie_stats[
        df_goalie_stats["goalie_id"].astype(str) == str(goalie_id)
    ]
    if goalie_rows.empty:
        return None

    merged = goalie_rows.merge(
        df_games[["game_id", "game_date"]],
        on="game_id",
        how="left",
    )
    prior = merged[merged["game_date"] < game_date].sort_values("game_date").tail(window)

    if len(prior) < MIN_GOALIE_STARTS:
        return None

    total_shots = prior["shots_against"].sum()
    if total_shots == 0:
        return None

    # Per-start sv% then EWMA
    sv_pct_series = prior.apply(
        lambda r: r["saves"] / r["shots_against"] if r["shots_against"] > 0 else LEAGUE_AVG_SV_PCT,
        axis=1,
    )
    sv_pct = _ewma(sv_pct_series)
    gsaa   = (sv_pct - LEAGUE_AVG_SV_PCT) * total_shots

    return {
        "sv_pct": round(sv_pct, 4),
        "gsaa":   round(gsaa, 3),
    }


# ── Rest days ─────────────────────────────────────────────────────────────────

def get_rest_days(
    team: str,
    game_date: str,
    df_games: pd.DataFrame,
    cap: int = 7,
) -> int:
    prior = df_games[
        ((df_games["home_team"] == team) | (df_games["away_team"] == team)) &
        (df_games["game_date"] < game_date)
    ].sort_values("game_date")

    if prior.empty:
        return cap

    last_date = datetime.strptime(prior.iloc[-1]["game_date"], "%Y-%m-%d")
    this_date = datetime.strptime(game_date, "%Y-%m-%d")
    days = (this_date - last_date).days
    return min(days, cap)
