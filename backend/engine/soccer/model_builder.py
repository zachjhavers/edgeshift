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
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

from db import get_conn, setup_db
from utils import DC_BASE_GOALS, HOST_ADVANTAGE, HOST_NATIONS, importance_weight

MODEL_PATH = Path(__file__).parent / "models" / "dc_model.pkl"


# ── Dixon-Coles optimizer ──────────────────────────────────────────────────────

def _log_likelihood(params: np.ndarray, teams: list[str],
                    h_idx: np.ndarray, a_idx: np.ndarray,
                    hg: np.ndarray, ag: np.ndarray,
                    weights: np.ndarray,
                    ha_mask: np.ndarray,
                    log_fac_hg: np.ndarray,
                    log_fac_ag: np.ndarray) -> float:
    """Vectorized Dixon-Coles negative log-likelihood."""
    n     = len(teams)
    alpha = np.exp(params[:n])
    delta = np.exp(params[n:2*n])
    rho   = params[2*n]
    base  = np.exp(params[2*n + 1])

    lam = base * alpha[h_idx] * delta[a_idx] * np.exp(HOST_ADVANTAGE * ha_mask)
    mu  = base * alpha[a_idx] * delta[h_idx]

    # Poisson log-PMF: k*log(λ) - λ - log(k!)  — gammaln(k+1) = log(k!)
    log_p_hg = hg * np.log(lam) - lam - log_fac_hg
    log_p_ag = ag * np.log(mu)  - mu  - log_fac_ag

    # Dixon-Coles low-score correction (only affects scores 0 or 1)
    tau = np.ones(len(hg))
    mask00 = (hg == 0) & (ag == 0); tau[mask00] = 1 - lam[mask00] * mu[mask00] * rho
    mask10 = (hg == 1) & (ag == 0); tau[mask10] = 1 + mu[mask10] * rho
    mask01 = (hg == 0) & (ag == 1); tau[mask01] = 1 + lam[mask01] * rho
    mask11 = (hg == 1) & (ag == 1); tau[mask11] = 1 - rho

    tau = np.clip(tau, 1e-10, None)
    log_lik = weights * (log_p_hg + log_p_ag + np.log(tau))

    if not np.isfinite(log_lik).all():
        return 1e9
    return -log_lik.sum()


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

    home_list, away_list, hg_list, ag_list, w_list, ha_list = [], [], [], [], [], []
    for r in rows:
        tw = _time_weight(r["match_date"], today)
        iw = importance_weight(r["tournament"] or "Friendly")
        home_list.append(r["home_team"])
        away_list.append(r["away_team"])
        hg_list.append(r["home_score"])
        ag_list.append(r["away_score"])
        w_list.append(tw * iw)
        # Host advantage: non-neutral match OR host nation at WC
        ha_list.append(1.0 if (not r["neutral"] or r["home_team"] in HOST_NATIONS) else 0.0)

    # Only keep teams that appear in at least 3 matches (prune tiny nations)
    from collections import Counter
    counts = Counter(home_list + away_list)
    valid  = {t for t, c in counts.items() if c >= 3}
    mask   = [i for i, (h, a) in enumerate(zip(home_list, away_list))
              if h in valid and a in valid]
    home_list = [home_list[i] for i in mask]
    away_list = [away_list[i] for i in mask]
    hg_list   = [hg_list[i] for i in mask]
    ag_list   = [ag_list[i] for i in mask]
    w_list    = [w_list[i] for i in mask]
    ha_list   = [ha_list[i] for i in mask]

    all_teams = sorted(valid)
    n         = len(all_teams)
    team_idx  = {t: i for i, t in enumerate(all_teams)}

    # Pre-compute numpy arrays for vectorized likelihood
    h_idx_arr = np.array([team_idx[h] for h in home_list], dtype=np.int32)
    a_idx_arr = np.array([team_idx[a] for a in away_list], dtype=np.int32)
    hg_arr    = np.array(hg_list, dtype=np.float64)
    ag_arr    = np.array(ag_list, dtype=np.float64)
    w_arr     = np.array(w_list,  dtype=np.float64)
    ha_arr    = np.array(ha_list, dtype=np.float64)

    print(f"  Fitting Dixon-Coles on {len(mask)} matches, {n} teams...")

    # Precompute log(k!) once — reused every likelihood call
    log_fac_hg = gammaln(hg_arr + 1)
    log_fac_ag = gammaln(ag_arr + 1)

    x0 = np.zeros(2 * n + 2)
    x0[2*n]     = -0.1
    x0[2*n + 1] = np.log(DC_BASE_GOALS)

    bounds = [(-3, 3)] * (2 * n) + [(-0.5, 0.0), (-1.0, 1.0)]

    result = minimize(
        _log_likelihood,
        x0,
        args=(all_teams, h_idx_arr, a_idx_arr, hg_arr, ag_arr, w_arr, ha_arr,
              log_fac_hg, log_fac_ag),
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
