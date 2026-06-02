"""
NHL EV engine — load today's predictions and market odds, compute expected
value and Kelly stakes, write positive-EV bets to the DB.

Market baseline (in priority order):
  1. Pinnacle vig-free probability  (sharpest market, ~2-4% vig)
  2. Consensus vig-free probability (average across all available books)
  3. DraftKings single-book         (fallback when Pinnacle absent)

EV is calculated against the best available odds across all books.
Line movement filter: if the pre-game Pinnacle line moved >=3pp against
our side vs the morning opening, the bet is vetoed.
"""

import os
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(os.getenv("NHL_DB_PATH", str(Path(__file__).parent / "nhl_predictor.db")))

STAKE             = 100.0
EV_THRESHOLD      = 5.0
MIN_MARKET_EDGE   = 0.04
MAX_MARKET_EDGE   = 0.07
LINE_MOVE_VETO_PP = 0.03   # veto if line moved >=3pp against our side


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _vig_free_prob(home_odds: float, away_odds: float) -> float:
    raw_h = 1.0 / home_odds
    raw_a = 1.0 / away_odds
    total = raw_h + raw_a
    return raw_h / total


def _kelly_pct(prob: float, decimal_odds: float) -> float:
    b = decimal_odds - 1.0
    q = 1.0 - prob
    raw = (b * prob - q) / b if b > 0 else 0.0
    return min(max(raw * 0.25, 0.0), 0.05) * 100.0


def _ensure_ev_table():
    conn = _conn()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS nhl_ev_bets (
            id                     INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date              TEXT NOT NULL,
            matchup                TEXT NOT NULL,
            side                   TEXT NOT NULL,
            team                   TEXT NOT NULL,
            model_prob             REAL NOT NULL,
            market_prob            REAL NOT NULL,
            pinnacle_prob          REAL,
            edge_vs_market         REAL NOT NULL,
            odds                   REAL NOT NULL,
            entry_book             TEXT,
            ev                     REAL NOT NULL,
            kelly_pct              REAL NOT NULL,
            line_move_direction    INTEGER DEFAULT 0,
            closing_pinnacle_odds  REAL,
            clv_pct                REAL,
            result                 TEXT DEFAULT 'TBD',
            created_at             TEXT NOT NULL,
            UNIQUE(game_date, matchup, side)
        )
    """)
    conn.commit()

    new_cols = [
        ("pinnacle_prob",         "REAL"),
        ("entry_book",            "TEXT"),
        ("line_move_direction",   "INTEGER DEFAULT 0"),
        ("closing_pinnacle_odds", "REAL"),
        ("clv_pct",               "REAL"),
    ]
    for col, dtype in new_cols:
        try:
            conn.execute(f"ALTER TABLE nhl_ev_bets ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.close()


def run_ev(as_of_date: str | None = None) -> list[dict]:
    today = as_of_date or datetime.now().strftime("%Y-%m-%d")
    print(f"--- NHL EV Engine  |  {today} ---")

    conn = _conn()

    preds = conn.execute("""
        SELECT p.game_id, p.home_win_prob, p.away_win_prob,
               g.home_team, g.away_team, g.game_date
        FROM predictions p
        JOIN games g ON g.game_id = p.game_id
        WHERE g.game_date = ?
    """, (today,)).fetchall()

    if not preds:
        print("  No predictions found for today — run predict.py first.")
        conn.close()
        return []

    odds_rows = conn.execute("""
        SELECT home_team, away_team,
               home_odds, away_odds,
               pinnacle_home_odds, pinnacle_away_odds,
               best_home_odds, best_away_odds, best_home_book, best_away_book,
               consensus_home_prob, opening_pinnacle_home_prob
        FROM historical_odds
        WHERE game_date BETWEEN date(?, '-1 day') AND date(?, '+1 day')
    """, (today, today)).fetchall()
    conn.close()

    odds_lookup: dict[tuple, dict] = {
        (r["home_team"], r["away_team"]): dict(r) for r in odds_rows
    }

    ev_bets: list[dict] = []
    now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    _ensure_ev_table()
    conn = _conn()

    for p in preds:
        home    = p["home_team"]
        away    = p["away_team"]
        prob_h  = float(p["home_win_prob"])
        prob_a  = float(p["away_win_prob"])
        matchup = f"{away} @ {home}"
        o       = odds_lookup.get((home, away))

        if o is None:
            print(f"  No odds for {matchup} — skipping EV.")
            continue

        # ── Market baseline ───────────────────────────────────────────────────
        pin_h = o["pinnacle_home_odds"]
        pin_a = o["pinnacle_away_odds"]
        if pin_h and pin_a and float(pin_h) > 1.0 and float(pin_a) > 1.0:
            mkt_home_prob = _vig_free_prob(float(pin_h), float(pin_a))
        elif o["consensus_home_prob"] is not None:
            mkt_home_prob = float(o["consensus_home_prob"])
        else:
            mkt_home_prob = _vig_free_prob(float(o["home_odds"]), float(o["away_odds"]))

        mkt_away_prob = 1.0 - mkt_home_prob

        # ── Best available odds ───────────────────────────────────────────────
        best_h_odds = float(o["best_home_odds"]) if o["best_home_odds"] else float(o["home_odds"])
        best_a_odds = float(o["best_away_odds"]) if o["best_away_odds"] else float(o["away_odds"])
        best_h_book = o["best_home_book"] or "draftkings"
        best_a_book = o["best_away_book"] or "draftkings"

        # ── Pinnacle prob for display ─────────────────────────────────────────
        pin_prob_h = _vig_free_prob(float(pin_h), float(pin_a)) if pin_h and pin_a else None

        opening = o["opening_pinnacle_home_prob"]

        for side, prob, dec_odds, book_key, mkt_prob in [
            ("home", prob_h, best_h_odds, best_h_book, mkt_home_prob),
            ("away", prob_a, best_a_odds, best_a_book, mkt_away_prob),
        ]:
            # ── Line movement ─────────────────────────────────────────────────
            line_move_direction = 0
            if opening is not None and pin_prob_h is not None:
                delta_h        = float(pin_prob_h) - float(opening)  # +ve = moved toward home
                delta_for_side = delta_h if side == "home" else -delta_h
                if delta_for_side >= 0.015:
                    line_move_direction = 1    # confirming
                elif delta_for_side <= -0.015:
                    line_move_direction = -1   # against

                if delta_for_side <= -LINE_MOVE_VETO_PP:
                    print(f"  VETO {matchup} {side}: line moved {delta_for_side*100:+.1f}pp against.")
                    continue

            edge = prob - mkt_prob
            ev   = (prob * (dec_odds - 1) * STAKE) - ((1 - prob) * STAKE)

            if ev >= EV_THRESHOLD and MIN_MARKET_EDGE <= edge <= MAX_MARKET_EDGE:
                team = home if side == "home" else away
                if pin_prob_h is not None:
                    pinnacle_prob_display = round(float(pin_prob_h), 4) if side == "home" else round(1.0 - float(pin_prob_h), 4)
                else:
                    pinnacle_prob_display = None

                bet = {
                    "game_date":           today,
                    "matchup":             matchup,
                    "side":                side,
                    "team":                team,
                    "model_prob":          round(prob, 4),
                    "market_prob":         round(mkt_prob, 4),
                    "pinnacle_prob":       pinnacle_prob_display,
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
                    INSERT INTO nhl_ev_bets
                        (game_date, matchup, side, team, model_prob, market_prob,
                         pinnacle_prob, edge_vs_market, odds, entry_book, ev, kelly_pct,
                         line_move_direction, result, created_at)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
                    bet["model_prob"], bet["market_prob"], bet["pinnacle_prob"],
                    bet["edge_vs_market"], bet["odds"], bet["entry_book"],
                    bet["ev"], bet["kelly_pct"], bet["line_move_direction"],
                    bet["result"], bet["created_at"],
                ))

    conn.commit()
    conn.close()

    ev_bets.sort(key=lambda x: x["ev"], reverse=True)

    if ev_bets:
        lm_icon = {1: "↑", -1: "↓", 0: "—"}
        print(f"\n{'='*78}")
        print(f"  NHL +EV BETS  (EV >${EV_THRESHOLD:.0f}/100  |  edge {MIN_MARKET_EDGE*100:.0f}–{MAX_MARKET_EDGE*100:.0f}pp  |  Pinnacle baseline)")
        print(f"{'='*78}\n")
        print(f"  {'Team':<22} {'Side':<5} {'Odds':>6}  {'Book':<12}  {'Model%':>6}  {'Mkt%':>6}  {'Edge':>7}  {'EV/100':>8}  {'Kelly%':>6}  LM")
        print(f"  {'-'*78}")
        for b in ev_bets:
            book_short = (b["entry_book"] or "")[:10]
            print(
                f"  {b['team']:<22} {b['side']:<5} "
                f"{b['odds']:>6.3f}  "
                f"{book_short:<12}  "
                f"{b['model_prob']*100:5.1f}%  "
                f"{b['market_prob']*100:5.1f}%  "
                f"{b['edge_vs_market']*100:+5.1f}pp  "
                f"${b['ev']:>7.2f}  "
                f"{b['kelly_pct']:>5.2f}%  "
                f"{lm_icon.get(b['line_move_direction'], '—')}"
            )
        print(f"\n  {len(ev_bets)} bet(s) written.")
    else:
        print(f"\n  No +EV bets today.")

    return ev_bets


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parents[2] / ".env")
    run_ev()
