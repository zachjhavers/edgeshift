"""
Fetch final scores for pending MLB predictions and update the database.
Also resolves CLV (Closing Line Value) for EV bets by fetching the
closing Pinnacle line from historical_odds at result time.

Safe to call repeatedly — only updates rows still at TBD.

Run standalone: python update_results.py
"""

import csv
import os
from datetime import datetime, timedelta

import requests
from sqlalchemy import text

from db import get_engine
from utils import MLB_TEAM_MAP

_CSV_PATH = os.path.join(os.path.dirname(__file__), "bets_log.csv")


def _ensure_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mlb_predictions (
                id                       SERIAL PRIMARY KEY,
                game_date                DATE        NOT NULL,
                matchup                  VARCHAR(60) NOT NULL,
                home_team                VARCHAR(10) NOT NULL,
                away_team                VARCHAR(10) NOT NULL,
                home_model_prob          FLOAT,
                away_model_prob          FLOAT,
                confidence               FLOAT,
                home_elo_prob            FLOAT,
                home_odds                FLOAT,
                away_odds                FLOAT,
                market_implied_home_prob FLOAT,
                ev_home                  FLOAT,
                ev_away                  FLOAT,
                result                   VARCHAR(20) NOT NULL DEFAULT 'TBD',
                created_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at               TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (game_date, home_team, away_team)
            )
        """))


def _ensure_ev_bets_table(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS mlb_ev_bets (
                id                    SERIAL PRIMARY KEY,
                game_date             DATE        NOT NULL,
                matchup               VARCHAR(60) NOT NULL,
                side                  VARCHAR(5)  NOT NULL,
                team                  VARCHAR(10) NOT NULL,
                model_prob            FLOAT,
                market_prob           FLOAT,
                pinnacle_prob         FLOAT,
                edge_vs_market        FLOAT,
                entry_odds            FLOAT,
                entry_book            VARCHAR(20),
                ev                    FLOAT,
                kelly_pct             FLOAT,
                line_move_direction   INTEGER,
                closing_pinnacle_odds FLOAT,
                clv_pct               FLOAT,
                result                VARCHAR(20) NOT NULL DEFAULT 'TBD',
                created_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                updated_at            TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                UNIQUE (game_date, matchup, side)
            )
        """))


def fetch_final_scores(date: str) -> dict:
    """Return {(home_abbr, away_abbr): (home_score, away_score)} for completed games."""
    url = (
        f"https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&date={date}&hydrate=linescore"
    )
    try:
        resp = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception as e:
        print(f"  Warning: could not fetch scores for {date} — {e}")
        return {}

    scores = {}
    for date_block in resp.json().get("dates", []):
        for game in date_block.get("games", []):
            state = game.get("status", {}).get("codedGameState", "")
            if state not in ("F", "FR"):
                continue
            teams      = game.get("teams", {})
            home_name  = teams.get("home", {}).get("team", {}).get("name", "")
            away_name  = teams.get("away", {}).get("team", {}).get("name", "")
            home_abbr  = MLB_TEAM_MAP.get(home_name, "")
            away_abbr  = MLB_TEAM_MAP.get(away_name, "")
            home_score = teams.get("home", {}).get("score")
            away_score = teams.get("away", {}).get("score")
            if home_abbr and away_abbr and home_score is not None and away_score is not None:
                scores[(home_abbr, away_abbr)] = (int(home_score), int(away_score))
    return scores


def _resolve_clv(engine, resolved: dict[tuple, str]) -> None:
    """
    For each just-resolved game, compute CLV for any EV bets placed on it.
    CLV% = (entry_odds / closing_pinnacle_odds) - 1
    Closing Pinnacle odds come from historical_odds at the last-fetched line.
    """
    if not resolved:
        return

    try:
        with engine.begin() as conn:
            for (date_str, home_team, away_team), result in resolved.items():
                matchup = f"% @ {home_team}"

                # Fetch closing Pinnacle odds for this game
                row = conn.execute(text("""
                    SELECT pinnacle_home_odds, pinnacle_away_odds
                    FROM historical_odds
                    WHERE game_date = :d AND home_team = :h AND away_team = :a
                """), {"d": date_str, "h": home_team, "a": away_team}).fetchone()

                if row is None or row[0] is None or row[1] is None:
                    continue

                pin_home_close = float(row[0])
                pin_away_close = float(row[1])

                # Update EV bets: set result + closing odds + CLV
                bets = conn.execute(text("""
                    SELECT id, side, entry_odds
                    FROM mlb_ev_bets
                    WHERE game_date = :d
                      AND matchup LIKE :matchup
                      AND result = 'TBD'
                """), {"d": date_str, "matchup": matchup}).fetchall()

                for bet in bets:
                    closing_odds = pin_home_close if bet.side == "home" else pin_away_close
                    clv_pct = round((float(bet.entry_odds) / closing_odds) - 1, 4) if closing_odds > 1.0 else None
                    conn.execute(text("""
                        UPDATE mlb_ev_bets
                        SET result = :result,
                            closing_pinnacle_odds = :closing_odds,
                            clv_pct = :clv_pct,
                            updated_at = NOW()
                        WHERE id = :id
                    """), {
                        "result":       result,
                        "closing_odds": closing_odds,
                        "clv_pct":      clv_pct,
                        "id":           bet.id,
                    })

        print(f"  CLV resolved for {len(resolved)} game(s).")
    except Exception as e:
        print(f"  Warning: CLV resolution failed — {e}")


def update_results():
    engine = get_engine()
    _ensure_table(engine)
    _ensure_ev_bets_table(engine)

    today = datetime.now().strftime("%Y-%m-%d")

    with engine.connect() as conn:
        pending = conn.execute(text("""
            SELECT id, game_date::text, home_team, away_team
            FROM mlb_predictions
            WHERE result = 'TBD'
              AND game_date <= :today
            ORDER BY game_date
        """), {"today": today}).fetchall()

    if not pending:
        print("No pending predictions to update.")
        _sync_csv_from_db(engine)
        return

    by_date: dict[str, list] = {}
    for row in pending:
        by_date.setdefault(row.game_date, []).append(row)

    total_updated = 0
    resolved: dict[tuple, str] = {}
    with engine.connect() as conn:
        for date_str, rows in by_date.items():
            scores = fetch_final_scores(date_str)
            if not scores:
                print(f"  {date_str}: no final scores available yet.")
                continue

            for row in rows:
                key = (row.home_team, row.away_team)
                if key not in scores:
                    continue
                home_score, away_score = scores[key]
                result = "HOME_WIN" if home_score > away_score else "AWAY_WIN"
                conn.execute(text("""
                    UPDATE mlb_predictions
                    SET result = :result, updated_at = NOW()
                    WHERE id = :id
                """), {"result": result, "id": row.id})
                resolved[(date_str, row.home_team, row.away_team)] = result
                total_updated += 1
                print(f"  {date_str} {row.away_team} @ {row.home_team}: "
                      f"{away_score}–{home_score} → {result}")

        conn.commit()

    print(f"Results updated: {total_updated} game(s).")

    _resolve_clv(engine, resolved)
    _sync_csv_from_db(engine)


def _sync_csv_from_db(engine):
    if not os.path.exists(_CSV_PATH):
        return
    try:
        with engine.connect() as conn:
            rows_pg = conn.execute(text("""
                SELECT game_date::text, home_team, away_team, result
                FROM mlb_predictions
                WHERE result != 'TBD'
            """)).fetchall()

        if not rows_pg:
            return

        pg_lookup = {(r.game_date, r.home_team, r.away_team): r.result for r in rows_pg}

        with open(_CSV_PATH, newline="") as f:
            reader    = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            csv_rows  = list(reader)

        updated = 0
        for row in csv_rows:
            key = (row.get("date", ""), row.get("home_team", ""), row.get("away_team", ""))
            if key in pg_lookup and row.get("result") == "TBD":
                row["result"] = pg_lookup[key]
                updated += 1

        with open(_CSV_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(csv_rows)

        if updated:
            print(f"  CSV synced: {updated} row(s) updated from PostgreSQL.")
    except Exception as e:
        print(f"  Warning: could not sync CSV from DB ({e})")


if __name__ == "__main__":
    print(f"--- Updating MLB Results ({datetime.now().strftime('%Y-%m-%d %H:%M')}) ---")
    update_results()
