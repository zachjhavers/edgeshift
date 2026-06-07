"""
Dixon-Coles Poisson model for international football.

Fits attack (α) and defense (δ) strength parameters per team using weighted
maximum likelihood estimation on historical match results.

Dixon-Coles low-score correction (ρ) adjusts joint probabilities for
(0,0), (1,0), (0,1), (1,1) outcomes which Poisson over-estimates.

Reference: Dixon & Coles (1997) "Modelling Association Football Scores
and Inefficiencies in the Football Betting Market"
"""

import pickle
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import factorial

from db import get_conn, setup_db
from utils import DC_BASE_GOALS, HOST_ADVANTAGE, HOST_NATIONS, importance_weight

MODEL_PATH = Path(__file__).parent / "models" / "dc_model.pkl"


# ── Dixon-Coles helpers ────────────────────────────────────────────────────────

def _tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles low-score correction factor."""
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
    return np.exp(-lam) * (lam ** k) / factorial(k)


def _log_likelihood(params: np.ndarray, teams: list[str],
                    home: list[str], away: list[str],
                    hg: list[int], ag: list[int],
                    weights: list[float],
                    neutral: list[int]) -> float:
    n = len(teams)
    alpha = np.exp(params[:n])      # attack strengths (positive)
    delta = np.exp(params[n:2*n])   # defense strengths (positive)
    rho   = params[2*n]             # low-score correction
    base  = np.exp(params[2*n + 1]) # base scoring rate

    team_idx = {t: i for i, t in enumerate(teams)}

    ll = 0.0
    for i in range(len(home)):
        h_idx = team_idx.get(home[i])
        a_idx = team_idx.get(away[i])
        if h_idx is None or a_idx is None:
            continue

        # Home advantage only for host nations at WC (all other games neutral)
        ha = HOST_ADVANTAGE if (neutral[i] == 0 or home[i] in HOST_NATIONS) else 0.0

        lam = base * alpha[h_idx] * delta[a_idx] * np.exp(ha)
        mu  = base * alpha[a_idx] * delta[h_idx]

        t = _tau(hg[i], ag[i], lam, mu, rho)
        if t <= 0:
            return np.inf

        p = t * _poisson_pmf(hg[i], lam) * _poisson_pmf(ag[i], mu)
        if p <= 0:
            return np.inf
        ll -= weights[i] * np.log(p)

    return ll


def _time_weight(date_str: str, today: str, half_life_days: float = 730.0) -> float:
    """Exponential time decay — matches 2 years old have weight 0.5."""
    try:
        delta = (datetime.strptime(today, "%Y-%m-%d") -
                 datetime.strptime(date_str, "%Y-%m-%d")).days
        return np.exp(-np.log(2) * delta / half_life_days)
    except Exception:
        return 0.5


# ── Training ───────────────────────────────────────────────────────────────────

def build_and_train_model(cutoff_date: str | None = None) -> dict:
    today = cutoff_date or datetime.now().strftime("%Y-%m-%d")

    conn = get_conn()
    rows = conn.execute("""
        SELECT match_date, home_team, away_team, home_score, away_score,
               tournament, neutral
        FROM matches
        WHERE result != 'TBD'
          AND home_score IS NOT NULL
        ORDER BY match_date
    """).fetchall()
    conn.close()

    if not rows:
        raise ValueError("No match data found. Run fetch_history.py first.")

    home_list, away_list, hg_list, ag_list, w_list, neu_list = [], [], [], [], [], []
    for r in rows:
        tw = _time_weight(r["match_date"], today)
        iw = importance_weight(r["tournament"] or "Friendly")
        home_list.append(r["home_team"])
        away_list.append(r["away_team"])
        hg_list.append(r["home_score"])
        ag_list.append(r["away_score"])
        w_list.append(tw * iw)
        neu_list.append(r["neutral"])

    all_teams = sorted(set(home_list) | set(away_list))
    n = len(all_teams)
    print(f"  Fitting Dixon-Coles on {len(rows)} matches, {n} teams...")

    # Initial params: log(1.0) = 0 for all, rho = -0.1, base = log(DC_BASE_GOALS)
    x0 = np.zeros(2 * n + 2)
    x0[2*n]     = -0.1
    x0[2*n + 1] = np.log(DC_BASE_GOALS)

    # Constraint: sum of log-attack = 0 (identifiability)
    bounds = [(-3, 3)] * (2 * n) + [(-0.5, 0.0), (-1.0, 1.0)]

    result = minimize(
        _log_likelihood,
        x0,
        args=(all_teams, home_list, away_list, hg_list, ag_list, w_list, neu_list),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-10},
    )

    params  = result.x
    alpha   = np.exp(params[:n])
    delta   = np.exp(params[n:2*n])
    rho     = float(params[2*n])
    base    = float(np.exp(params[2*n + 1]))

    team_params = {
        team: {"attack": float(alpha[i]), "defense": float(delta[i])}
        for i, team in enumerate(all_teams)
    }

    bundle = {
        "teams":       all_teams,
        "team_params": team_params,
        "rho":         rho,
        "base":        base,
        "train_date":  today,
        "log_lik":     float(-result.fun),
        "n_matches":   len(rows),
    }

    # Save
    MODEL_PATH.parent.mkdir(exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    # Persist params to DB
    conn = get_conn()
    for team, p in team_params.items():
        conn.execute("""
            INSERT INTO team_params (team, attack, defense)
            VALUES (?,?,?)
            ON CONFLICT(team) DO UPDATE SET
                attack     = excluded.attack,
                defense    = excluded.defense,
                updated_at = datetime('now')
        """, (team, p["attack"], p["defense"]))
    conn.execute("""
        INSERT INTO model_runs (run_date, n_teams, rho, log_lik)
        VALUES (?,?,?,?)
    """, (today, n, rho, bundle["log_lik"]))
    conn.commit()
    conn.close()

    print(f"  Done. ρ={rho:.4f}, base={base:.4f}, log-lik={bundle['log_lik']:.1f}")
    print(f"  Model saved to {MODEL_PATH}")
    return bundle


def load_model() -> dict | None:
    if not MODEL_PATH.exists():
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    setup_db()
    build_and_train_model()
