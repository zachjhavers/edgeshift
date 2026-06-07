"""
Resolve completed World Cup match results and mark EV bets as WIN/LOSS/DRAW_PUSH.
Pulls final scores from the ESPN unofficial API.
"""

import time
from datetime import datetime, timedelta

import requests

from db import get_conn
from utils import canonical

ESPN_BASE = "https://site.api.espn.com/apis/site/v2/sports/soccer"
WC_SLUG   = "fifa.world"


def _fetch_scores(date_str: str) -> dict[tuple, tuple]:
    """Return {(home, away): (home_score, away_score)} for completed matches."""
    espn_date = date_str.replace("-", "")
    try:
        r = requests.get(
            f"{ESPN_BASE}/{WC_SLUG}/scoreboard",
            params={"dates": espn_date},
            timeout=12,
        )
        r.raise_for_status()
        data = r.json()
        time.sleep(0.3)
    except Exception as e:
        print(f"  ESPN fetch failed for {date_str}: {e}")
        return {}

    scores = {}
    for event in data.get("events", []):
        comp   = event.get("competitions", [{}])[0]
        status = comp.get("status", {}).get("type", {}).get("state", "pre")
        if status != "post":
            continue

        competitors = comp.get("competitors", [])
        home_name = away_name = None
        home_score = away_score = None
        for c in competitors:
            name  = canonical(c.get("team", {}).get("displayName", ""))
            score = c.get("score")
            if c.get("homeAway") == "home":
                home_name  = name
                home_score = int(score) if score is not None else None
            else:
                away_name  = name
                away_score = int(score) if score is not None else None

        if home_name and away_name and home_score is not None and away_score is not None:
            scores[(home_name, away_name)] = (home_score, away_score)

    return scores


def _pending_dates() -> list[str]:
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    conn = get_conn()
    rows = conn.execute("""
        SELECT DISTINCT match_date FROM (
            SELECT match_date FROM matches WHERE result = 'TBD' AND match_date <= ?
            UNION
            SELECT match_date FROM soccer_ev_bets WHERE result = 'TBD' AND match_date <= ?
        ) ORDER BY match_date
    """, (yesterday, yesterday)).fetchall()
    conn.close()
    return [r[0] for r in rows]


def update_results() -> None:
    pending = _pending_dates()
    if not pending:
        print("  No pending soccer matches to resolve.")
        return

    print(f"  Checking {len(pending)} date(s): {pending[0]} → {pending[-1]}")
    conn = get_conn()

    for date_str in pending:
        scores = _fetch_scores(date_str)
        if not scores:
            continue

        for (home, away), (hs, as_) in scores.items():
            result = "HOME_WIN" if hs > as_ else ("AWAY_WIN" if as_ > hs else "DRAW")

            conn.execute("""
                UPDATE matches SET home_score=?, away_score=?, result=?
                WHERE match_date=? AND home_team=? AND away_team=? AND result='TBD'
            """, (hs, as_, result, date_str, home, away))

            # Resolve EV bets
            bets = conn.execute("""
                SELECT id, market, side FROM soccer_ev_bets
                WHERE match_date=? AND matchup=? AND result='TBD'
            """, (date_str, f"{home} vs {away}")).fetchall()

            for bet in bets:
                if bet["market"] == "h2h":
                    if bet["side"] == "home":
                        br = "WIN" if result == "HOME_WIN" else "LOSS"
                    elif bet["side"] == "draw":
                        br = "WIN" if result == "DRAW" else "LOSS"
                    else:
                        br = "WIN" if result == "AWAY_WIN" else "LOSS"
                elif bet["market"] == "totals":
                    # Need to know the line — fetch from bet label
                    row = conn.execute(
                        "SELECT label FROM soccer_ev_bets WHERE id=?", (bet["id"],)
                    ).fetchone()
                    total_goals = hs + as_
                    label = row["label"] if row else ""
                    try:
                        line = float(label.split()[-1])
                    except (ValueError, IndexError):
                        line = 2.5
                    if total_goals > line:
                        br = "WIN" if bet["side"] == "over" else "LOSS"
                    elif total_goals < line:
                        br = "WIN" if bet["side"] == "under" else "LOSS"
                    else:
                        br = "PUSH"
                else:
                    continue

                conn.execute(
                    "UPDATE soccer_ev_bets SET result=? WHERE id=?",
                    (br, bet["id"]),
                )
                print(f"  {date_str} {home} vs {away}: {hs}-{as_} → bet {bet['side']} = {br}")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    update_results()
