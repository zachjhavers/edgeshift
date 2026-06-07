"""
Soccer EV engine — compares model probabilities to Pinnacle vig-free lines.
Generates bets for:
  - 1X2 moneyline (home win / draw / away win)
  - Over/under 2.5 goals
"""

from datetime import datetime

from db import get_conn
from predict import generate_predictions, predict_match
from model_builder import load_model
from utils import (
    EV_THRESHOLD, MIN_MARKET_EDGE, MAX_MARKET_EDGE, MAX_KELLY, canonical
)


def _remove_vig_3way(h: float, d: float, a: float) -> tuple[float, float, float]:
    total = 1/h + 1/d + 1/a
    return 1/(h * total), 1/(d * total), 1/(a * total)


def _kelly(model_p: float, decimal_odds: float) -> float:
    b = decimal_odds - 1
    q = 1 - model_p
    k = (b * model_p - q) / b
    return max(0.0, min(k, MAX_KELLY))


def _ev(model_p: float, decimal_odds: float) -> float:
    return round((model_p * decimal_odds - 1) * 100, 2)


def run_ev(target_date: str | None = None) -> list[dict]:
    today = target_date or datetime.now().strftime("%Y-%m-%d")
    bundle = load_model()
    if bundle is None:
        print("  No model — run model_builder.py first.")
        return []

    conn = get_conn()

    # Get today's predictions — allow ±1 day odds window for UTC vs local date
    preds = conn.execute("""
        SELECT p.home_team, p.away_team,
               p.home_prob, p.draw_prob, p.away_prob,
               p.exp_home_goals, p.exp_away_goals,
               o.pinnacle_home_odds, o.pinnacle_draw_odds, o.pinnacle_away_odds,
               o.pinnacle_over_odds, o.pinnacle_under_odds, o.total_line
        FROM predictions p
        JOIN historical_odds o
          ON ABS(julianday(p.match_date) - julianday(o.match_date)) <= 1
         AND p.home_team  = o.home_team
         AND p.away_team  = o.away_team
        WHERE p.match_date = ?
    """, (today,)).fetchall()

    if not preds:
        print(f"  No predictions with odds for {today}.")
        conn.close()
        return []

    # Clear stale bets for today
    conn.execute("DELETE FROM soccer_ev_bets WHERE match_date = ?", (today,))
    conn.commit()

    bets = []
    for p in preds:
        matchup = f"{p['home_team']} vs {p['away_team']}"

        ph, pd_, pa = float(p["home_prob"]), float(p["draw_prob"]), float(p["away_prob"])

        # ── 1X2 ───────────────────────────────────────────────────────────────
        if all([p["pinnacle_home_odds"], p["pinnacle_draw_odds"], p["pinnacle_away_odds"]]):
            pin_h, pin_d, pin_a = float(p["pinnacle_home_odds"]), float(p["pinnacle_draw_odds"]), float(p["pinnacle_away_odds"])
            mkt_h, mkt_d, mkt_a = _remove_vig_3way(pin_h, pin_d, pin_a)

            for label, side, model_p, mkt_p, pin_odds in [
                ("Home", "home", ph, mkt_h, pin_h),
                ("Draw", "draw", pd_, mkt_d, pin_d),
                ("Away", "away", pa, mkt_a, pin_a),
            ]:
                edge = model_p - mkt_p
                if edge < MIN_MARKET_EDGE or edge > MAX_MARKET_EDGE:
                    continue
                ev = _ev(model_p, pin_odds)
                if ev < EV_THRESHOLD:
                    continue
                kelly = _kelly(model_p, pin_odds)
                bets.append({
                    "match_date":     today,
                    "matchup":        matchup,
                    "market":         "h2h",
                    "side":           side,
                    "label":          label,
                    "model_prob":     round(model_p, 4),
                    "market_prob":    round(mkt_p, 4),
                    "pinnacle_prob":  round(mkt_p, 4),
                    "edge_vs_market": round(edge, 4),
                    "entry_odds":     pin_odds,
                    "entry_book":     "pinnacle",
                    "ev":             ev,
                    "kelly_pct":      round(kelly * 100, 2),
                })

        # ── Over/Under 2.5 goals ──────────────────────────────────────────────
        if p["pinnacle_over_odds"] and p["pinnacle_under_odds"] and p["total_line"]:
            pin_over  = float(p["pinnacle_over_odds"])
            pin_under = float(p["pinnacle_under_odds"])
            line      = float(p["total_line"])

            # Re-predict with correct line if not 2.5
            if abs(line - 2.5) > 0.01:
                from predict import predict_match as _pm
                fresh = _pm(p["home_team"], p["away_team"], bundle)
                p_over  = fresh.get("p_over_2_5", p["exp_home_goals"])
            else:
                p_over = float(p["exp_home_goals"])  # placeholder handled below

            # Compute p_over for the actual line using score matrix
            from predict import _score_matrix
            lam = bundle["base"] * bundle["team_params"].get(p["home_team"], {"attack": 1.0})["attack"] * \
                  bundle["team_params"].get(p["away_team"], {"defense": 1.0})["defense"]
            mu  = bundle["base"] * bundle["team_params"].get(p["away_team"], {"attack": 1.0})["attack"] * \
                  bundle["team_params"].get(p["home_team"], {"defense": 1.0})["defense"]
            from utils import HOST_NATIONS, HOST_ADVANTAGE
            import math
            if p["home_team"] in HOST_NATIONS:
                lam *= math.exp(HOST_ADVANTAGE)
            mat = _score_matrix(lam, mu, bundle["rho"])
            import numpy as np
            p_over = float(sum(
                mat[i, j]
                for i in range(mat.shape[0])
                for j in range(mat.shape[1])
                if i + j > line
            ))
            p_under = 1 - p_over

            mkt_over, mkt_under = _remove_vig_3way(
                pin_over, pin_under, 999  # 2-way market: approximate
            )[:2]
            # Correct 2-way devig
            tot = 1/pin_over + 1/pin_under
            mkt_over  = (1/pin_over) / tot
            mkt_under = (1/pin_under) / tot

            for label, side, model_p, mkt_p, pin_odds in [
                (f"Over {line}", "over",  p_over,  mkt_over,  pin_over),
                (f"Under {line}", "under", p_under, mkt_under, pin_under),
            ]:
                edge = model_p - mkt_p
                if edge < MIN_MARKET_EDGE or edge > MAX_MARKET_EDGE:
                    continue
                ev = _ev(model_p, pin_odds)
                if ev < EV_THRESHOLD:
                    continue
                kelly = _kelly(model_p, pin_odds)
                bets.append({
                    "match_date":     today,
                    "matchup":        matchup,
                    "market":         "totals",
                    "side":           side,
                    "label":          label,
                    "model_prob":     round(model_p, 4),
                    "market_prob":    round(mkt_p, 4),
                    "pinnacle_prob":  round(mkt_p, 4),
                    "edge_vs_market": round(edge, 4),
                    "entry_odds":     pin_odds,
                    "entry_book":     "pinnacle",
                    "ev":             ev,
                    "kelly_pct":      round(kelly * 100, 2),
                })

    # Write to DB
    for bet in bets:
        conn.execute("""
            INSERT INTO soccer_ev_bets
                (match_date, matchup, market, side, label,
                 model_prob, market_prob, pinnacle_prob, edge_vs_market,
                 entry_odds, entry_book, ev, kelly_pct)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(match_date, matchup, market, side) DO UPDATE SET
                model_prob      = excluded.model_prob,
                market_prob     = excluded.market_prob,
                pinnacle_prob   = excluded.pinnacle_prob,
                edge_vs_market  = excluded.edge_vs_market,
                entry_odds      = excluded.entry_odds,
                ev              = excluded.ev,
                kelly_pct       = excluded.kelly_pct
        """, (bet["match_date"], bet["matchup"], bet["market"], bet["side"],
              bet["label"], bet["model_prob"], bet["market_prob"],
              bet["pinnacle_prob"], bet["edge_vs_market"],
              bet["entry_odds"], bet["entry_book"], bet["ev"], bet["kelly_pct"]))
    conn.commit()

    print(f"\n--- Soccer EV Engine  |  {today} ---")
    if bets:
        print(f"  {'Matchup':<30} {'Side':<12} {'Book':<10} {'Odds':>6} "
              f"{'Model%':>7} {'Mkt%':>6} {'Edge':>7} {'EV/100':>8} {'Kelly%':>7}")
        print(f"  {'-'*95}")
        for b in sorted(bets, key=lambda x: -x["ev"]):
            print(f"  {b['matchup']:<30} {b['label']:<12} {b['entry_book']:<10} "
                  f"{b['entry_odds']:>6.3f} {b['model_prob']*100:>6.1f}% "
                  f"{b['market_prob']*100:>5.1f}% +{b['edge_vs_market']*100:>4.1f}pp "
                  f"${b['ev']:>7.2f} {b['kelly_pct']:>6.2f}%")
        print(f"\n  {len(bets)} bet(s) logged.")
    else:
        print("  No +EV soccer bets today.")

    conn.close()
    return bets


if __name__ == "__main__":
    from db import setup_db
    setup_db()
    run_ev()
