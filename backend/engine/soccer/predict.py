"""
Generate match predictions for today's WC fixtures using the Dixon-Coles model.
Outputs P(home win), P(draw), P(away win) and expected goals for each match.
"""

import math
from datetime import datetime

import numpy as np
from scipy.special import factorial

from db import get_conn
from model_builder import load_model
from utils import HOST_ADVANTAGE, HOST_NATIONS, canonical

MAX_GOALS = 10   # truncate Poisson at 10 goals per side


def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


def _poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return float(k == 0)
    return float(np.exp(-lam) * (lam ** k) / factorial(k))


def _score_matrix(lam: float, mu: float, rho: float) -> np.ndarray:
    """Build (MAX_GOALS+1) x (MAX_GOALS+1) joint probability matrix."""
    mat = np.zeros((MAX_GOALS + 1, MAX_GOALS + 1))
    for i in range(MAX_GOALS + 1):
        for j in range(MAX_GOALS + 1):
            t   = _tau(i, j, lam, mu, rho)
            mat[i, j] = t * _poisson_pmf(i, lam) * _poisson_pmf(j, mu)
    # Renormalise to correct for truncation
    total = mat.sum()
    if total > 0:
        mat /= total
    return mat


def _probs_from_matrix(mat: np.ndarray) -> dict:
    home_win = float(np.sum(np.tril(mat, -1)))   # home scores more
    draw     = float(np.sum(np.diag(mat)))
    away_win = float(np.sum(np.triu(mat, 1)))

    total = home_win + draw + away_win
    if total > 0:
        home_win /= total
        draw     /= total
        away_win /= total

    # Expected goals
    goals = np.arange(MAX_GOALS + 1)
    exp_home = float(np.sum(mat.sum(axis=1) * goals))
    exp_away = float(np.sum(mat.sum(axis=0) * goals))

    # P(over 2.5 goals)
    p_over_2_5 = float(np.sum(mat[i, j]
                               for i in range(MAX_GOALS + 1)
                               for j in range(MAX_GOALS + 1)
                               if i + j > 2))

    return {
        "home_prob":    round(home_win, 4),
        "draw_prob":    round(draw, 4),
        "away_prob":    round(away_win, 4),
        "exp_home_goals": round(exp_home, 3),
        "exp_away_goals": round(exp_away, 3),
        "p_over_2_5":   round(p_over_2_5, 4),
        "p_under_2_5":  round(1 - p_over_2_5, 4),
    }


def _fallback_params(bundle: dict, team: str) -> dict:
    """For unknown teams return league-average params (attack=1, defense=1)."""
    return bundle["team_params"].get(team, {"attack": 1.0, "defense": 1.0})


def predict_match(home_team: str, away_team: str,
                  bundle: dict, neutral: bool = True) -> dict:
    hp = _fallback_params(bundle, home_team)
    ap = _fallback_params(bundle, away_team)

    # Home advantage only for WC host nations
    ha = HOST_ADVANTAGE if (not neutral or home_team in HOST_NATIONS) else 0.0

    lam = bundle["base"] * hp["attack"] * ap["defense"] * math.exp(ha)
    mu  = bundle["base"] * ap["attack"] * hp["defense"]

    mat = _score_matrix(lam, mu, bundle["rho"])
    return _probs_from_matrix(mat) | {"exp_lambda": round(lam, 3), "exp_mu": round(mu, 3)}


def generate_predictions(target_date: str | None = None) -> list[dict]:
    bundle = load_model()
    if bundle is None:
        print("No model found. Run model_builder.py first.")
        return []

    today = target_date or datetime.now().strftime("%Y-%m-%d")
    conn  = get_conn()
    games = conn.execute("""
        SELECT match_date, home_team, away_team, neutral
        FROM matches
        WHERE match_date = ? AND result = 'TBD'
        ORDER BY match_date
    """, (today,)).fetchall()
    conn.close()

    if not games:
        print(f"  No scheduled WC matches for {today}.")
        return []

    results = []
    conn = get_conn()
    for g in games:
        pred = predict_match(g["home_team"], g["away_team"], bundle,
                             neutral=bool(g["neutral"]))
        conn.execute("""
            INSERT INTO predictions
                (match_date, home_team, away_team,
                 home_prob, draw_prob, away_prob,
                 exp_home_goals, exp_away_goals)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(match_date, home_team, away_team) DO UPDATE SET
                home_prob      = excluded.home_prob,
                draw_prob      = excluded.draw_prob,
                away_prob      = excluded.away_prob,
                exp_home_goals = excluded.exp_home_goals,
                exp_away_goals = excluded.exp_away_goals,
                created_at     = datetime('now')
        """, (today, g["home_team"], g["away_team"],
              pred["home_prob"], pred["draw_prob"], pred["away_prob"],
              pred["exp_home_goals"], pred["exp_away_goals"]))
        results.append({
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            **pred,
        })

    conn.commit()
    conn.close()

    # ── Print report ──────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  FIFA WC Predictions  |  {today}")
    print(f"{'='*65}")
    print(f"  {'Matchup':<30} {'Home%':>6} {'Draw%':>6} {'Away%':>6}  xG")
    print(f"  {'-'*62}")
    for r in results:
        matchup = f"{r['home_team']} vs {r['away_team']}"
        print(f"  {matchup:<30} {r['home_prob']*100:>5.1f}% {r['draw_prob']*100:>5.1f}% "
              f"{r['away_prob']*100:>5.1f}%  {r['exp_home_goals']:.2f}-{r['exp_away_goals']:.2f}")
    print(f"{'='*65}\n")

    return results


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    generate_predictions()
