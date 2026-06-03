"""
NBA EV engine — compute expected value and Kelly stakes for today's games.
Uses separate regular/playoff models. Pinnacle vig-free prob is the market
baseline; best US book odds are used for EV calculation and display.
"""

import os
from datetime import datetime
from pathlib import Path

from db import get_conn

DB_PATH = Path(os.getenv("NBA_DB_PATH", str(Path(__file__).parent / "nba_predictor.db")))

STAKE             = 100.0
EV_THRESHOLD      = 5.0
MIN_MARKET_EDGE   = 0.04
MAX_MARKET_EDGE   = 0.07
LINE_MOVE_VETO_PP = 0.03


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    raw_h = 1.0 / home_odds
    raw_a = 1.0 / away_odds
    return raw_h / (raw_h + raw_a)


def _kelly_pct(prob: float, decimal_odds: float) -> float:
    b   = decimal_odds - 1.0
    q   = 1.0 - prob
    raw = (b * prob - q) / b if b > 0 else 0.0
    return min(max(raw * 0.25, 0.0), 0.05) * 100.0


def run_ev(as_of_date: str | None = None) -> list[dict]:
    today = as_of_date or datetime.now().strftime("%Y-%m-%d")
    print(f"--- NBA EV Engine  |  {today} ---")

    conn = get_conn()

    preds = conn.execute("""
        SELECT game_id, home_team, away_team, game_type,
               home_win_prob, away_win_prob
        FROM predictions
        WHERE game_date = ?
    """, (today,)).fetchall()

    if not preds:
        print("  No predictions for today — run predict.py first.")
        conn.close()
        return []

    odds_rows = conn.execute("""
        SELECT home_team, away_team,
               pinnacle_home_odds, pinnacle_away_odds,
               best_home_odds, best_away_odds, best_home_book, best_away_book,
               consensus_home_prob, opening_pinnacle_home_prob
        FROM historical_odds
        WHERE game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
    """, (today, today)).fetchall()
    conn.close()

    odds_lookup = {(r["home_team"], r["away_team"]): dict(r) for r in odds_rows}

    ev_bets: list[dict] = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    conn = get_conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nba_ev_bets (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date             TEXT NOT NULL,
            matchup               TEXT NOT NULL,
            side                  TEXT NOT NULL,
            team                  TEXT NOT NULL,
            game_type             TEXT NOT NULL,
            model_prob            REAL NOT NULL,
            market_prob           REAL NOT NULL,
            pinnacle_prob         REAL,
            edge_vs_market        REAL NOT NULL,
            odds                  REAL NOT NULL,
            entry_book            TEXT,
            ev                    REAL NOT NULL,
            kelly_pct             REAL NOT NULL,
            line_move_direction   INTEGER DEFAULT 0,
            closing_pinnacle_odds REAL,
            clv_pct               REAL,
            result                TEXT DEFAULT 'TBD',
            created_at            TEXT NOT NULL,
            UNIQUE(game_date, matchup, side)
        )
    """)

    for p in preds:
        home      = p["home_team"]
        away      = p["away_team"]
        prob_h    = float(p["home_win_prob"])
        prob_a    = float(p["away_win_prob"])
        game_type = p["game_type"]
        matchup   = f"{away} @ {home}"
        o         = odds_lookup.get((home, away))

        if o is None:
            print(f"  No odds for {matchup} — skipping.")
            continue

        # Market baseline: Pinnacle vig-free, then consensus, then display odds
        pin_h = o["pinnacle_home_odds"]
        pin_a = o["pinnacle_away_odds"]
        if pin_h and pin_a and float(pin_h) > 1.0 and float(pin_a) > 1.0:
            mkt_home_prob = _vig_free_prob(float(pin_h), float(pin_a))
        elif o["consensus_home_prob"] is not None:
            mkt_home_prob = float(o["consensus_home_prob"])
        else:
            continue

        mkt_away_prob = 1.0 - mkt_home_prob
        pin_prob_h    = _vig_free_prob(float(pin_h), float(pin_a)) if pin_h and pin_a else None
        opening       = o["opening_pinnacle_home_prob"]

        best_h_odds = float(o["best_home_odds"]) if o["best_home_odds"] else None
        best_a_odds = float(o["best_away_odds"]) if o["best_away_odds"] else None
        best_h_book = o["best_home_book"] or "draftkings"
        best_a_book = o["best_away_book"] or "draftkings"

        if not best_h_odds or not best_a_odds:
            continue

        for side, prob, dec_odds, book_key, mkt_prob in [
            ("home", prob_h, best_h_odds, best_h_book, mkt_home_prob),
            ("away", prob_a, best_a_odds, best_a_book, mkt_away_prob),
        ]:
            line_move_direction = 0
            if opening is not None and pin_prob_h is not None:
                delta_h        = float(pin_prob_h) - float(opening)
                delta_for_side = delta_h if side == "home" else -delta_h
                if delta_for_side >= 0.015:
                    line_move_direction = 1
                elif delta_for_side <= -0.015:
                    line_move_direction = -1
                if delta_for_side <= -LINE_MOVE_VETO_PP:
                    print(f"  VETO {matchup} {side}: line moved {delta_for_side*100:+.1f}pp against.")
                    continue

            edge = prob - mkt_prob
            ev   = (prob * (dec_odds - 1) * STAKE) - ((1 - prob) * STAKE)

            if ev >= EV_THRESHOLD and MIN_MARKET_EDGE <= edge <= MAX_MARKET_EDGE:
                team      = home if side == "home" else away
                pin_disp  = (round(float(pin_prob_h), 4) if side == "home"
                             else round(1.0 - float(pin_prob_h), 4)) if pin_prob_h else None

                bet = {
                    "game_date":           today,
                    "matchup":             matchup,
                    "side":                side,
                    "team":                team,
                    "game_type":           game_type,
                    "model_prob":          round(prob, 4),
                    "market_prob":         round(mkt_prob, 4),
                    "pinnacle_prob":       pin_disp,
                    "edge_vs_market":      round(edge, 4),
                    "odds":                round(dec_odds, 3),
                    "entry_book":          book_key,
                    "ev":                  round(ev, 2),
                    "kelly_pct":           round(_kelly_pct(prob, dec_odds), 2),
                    "line_move_direction": line_move_direction,
                    "result":              "TBD",
                    "created_at":          now,
                }
                ev_bets.append(bet)

                conn.execute("""
                    INSERT INTO nba_ev_bets
                        (game_date, matchup, side, team, game_type, model_prob, market_prob,
                         pinnacle_prob, edge_vs_market, odds, entry_book, ev, kelly_pct,
                         line_move_direction, result, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    ON CONFLICT(game_date, matchup, side) DO UPDATE SET
                        model_prob          = excluded.model_prob,
                        market_prob         = excluded.market_prob,
                        pinnacle_prob       = excluded.pinnacle_prob,
                        edge_vs_market      = excluded.edge_vs_market,
                        odds                = excluded.odds,
                        entry_book          = excluded.entry_book,
                        ev                  = excluded.ev,
                        kelly_pct           = excluded.kelly_pct,
                        line_move_direction = excluded.line_move_direction,
                        created_at          = excluded.created_at
                """, (
                    bet["game_date"], bet["matchup"], bet["side"], bet["team"],
                    bet["game_type"], bet["model_prob"], bet["market_prob"],
                    bet["pinnacle_prob"], bet["edge_vs_market"], bet["odds"],
                    bet["entry_book"], bet["ev"], bet["kelly_pct"],
                    bet["line_move_direction"], bet["result"], bet["created_at"],
                ))

    conn.commit()
    conn.close()

    ev_bets.sort(key=lambda x: x["ev"], reverse=True)

    lm_icon = {1: "↑", -1: "↓", 0: "—"}
    if ev_bets:
        print(f"\n{'='*82}")
        print(f"  NBA +EV BETS  (EV>${EV_THRESHOLD:.0f}/100  |  edge {MIN_MARKET_EDGE*100:.0f}–{MAX_MARKET_EDGE*100:.0f}pp  |  Pinnacle baseline)")
        print(f"{'='*82}\n")
        print(f"  {'Team':<20} {'Side':<5} {'Type':<8} {'Odds':>6}  {'Book':<12}  "
              f"{'Model%':>6}  {'Mkt%':>6}  {'Edge':>7}  {'EV/100':>8}  {'Kelly%':>6}  LM")
        print(f"  {'-'*82}")
        for b in ev_bets:
            print(
                f"  {b['team']:<20} {b['side']:<5} {b['game_type']:<8} "
                f"{b['odds']:>6.3f}  {(b['entry_book'] or ''):<12}  "
                f"{b['model_prob']*100:5.1f}%  {b['market_prob']*100:5.1f}%  "
                f"{b['edge_vs_market']*100:+5.1f}pp  ${b['ev']:>7.2f}  "
                f"{b['kelly_pct']:>5.2f}%  {lm_icon.get(b['line_move_direction'], '—')}"
            )
        print(f"\n  {len(ev_bets)} bet(s) written.")
    else:
        print(f"\n  No +EV NBA bets today.")

    return ev_bets


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    run_ev()
