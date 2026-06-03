"""
Resolve TBD NBA EV bets once game results are available.
Also computes Closing Line Value (CLV) against the final Pinnacle line.
"""

from datetime import datetime
from pathlib import Path

from db import get_conn


def update_results():
    conn = get_conn()

    pending = conn.execute("""
        SELECT id, game_date, matchup, side, team, odds
        FROM nba_ev_bets
        WHERE result = 'TBD'
    """).fetchall()

    if not pending:
        print("  No pending NBA bets to resolve.")
        conn.close()
        return

    resolved = 0
    for bet in pending:
        bet_id    = bet["id"]
        game_date = bet["game_date"]
        side      = bet["side"]

        # Parse matchup "AWAY @ HOME"
        parts = bet["matchup"].split(" @ ")
        if len(parts) != 2:
            continue
        away_team, home_team = parts[0].strip(), parts[1].strip()

        game = conn.execute("""
            SELECT home_win FROM games
            WHERE game_date = ? AND home_team = ? AND away_team = ?
              AND home_win IS NOT NULL
        """, (game_date, home_team, away_team)).fetchone()

        if game is None:
            continue

        home_win = game["home_win"]
        if side == "home":
            result = "WIN" if home_win == 1 else "LOSS"
        else:
            result = "WIN" if home_win == 0 else "LOSS"

        # CLV: compare entry odds vs closing Pinnacle odds
        clv_pct = None
        odds_row = conn.execute("""
            SELECT pinnacle_home_odds, pinnacle_away_odds
            FROM historical_odds
            WHERE game_date = ? AND home_team = ? AND away_team = ?
        """, (game_date, home_team, away_team)).fetchone()

        if odds_row:
            pin_h = odds_row["pinnacle_home_odds"]
            pin_a = odds_row["pinnacle_away_odds"]
            if pin_h and pin_a:
                closing_odds = float(pin_h) if side == "home" else float(pin_a)
                entry_odds   = float(bet["odds"])
                clv_pct      = round((entry_odds - closing_odds) / closing_odds * 100, 4)

                conn.execute("""
                    UPDATE nba_ev_bets
                    SET result = ?, closing_pinnacle_odds = ?, clv_pct = ?
                    WHERE id = ?
                """, (result, closing_odds, clv_pct, bet_id))
            else:
                conn.execute("UPDATE nba_ev_bets SET result = ? WHERE id = ?",
                             (result, bet_id))
        else:
            conn.execute("UPDATE nba_ev_bets SET result = ? WHERE id = ?",
                         (result, bet_id))

        resolved += 1

    conn.commit()
    conn.close()
    print(f"  Resolved {resolved} NBA bet(s).")


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    update_results()
