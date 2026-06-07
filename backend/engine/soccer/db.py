import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("SOCCER_DB_PATH", str(Path(__file__).parent / "soccer_predictor.db")))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS matches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date  TEXT NOT NULL,
            home_team   TEXT NOT NULL,
            away_team   TEXT NOT NULL,
            home_score  INTEGER,
            away_score  INTEGER,
            tournament  TEXT,
            neutral     INTEGER DEFAULT 1,
            result      TEXT DEFAULT 'TBD',
            UNIQUE(match_date, home_team, away_team)
        );

        CREATE TABLE IF NOT EXISTS team_params (
            team        TEXT PRIMARY KEY,
            attack      REAL NOT NULL DEFAULT 1.0,
            defense     REAL NOT NULL DEFAULT 1.0,
            updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS historical_odds (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date           TEXT NOT NULL,
            home_team            TEXT NOT NULL,
            away_team            TEXT NOT NULL,
            pinnacle_home_odds   REAL,
            pinnacle_draw_odds   REAL,
            pinnacle_away_odds   REAL,
            pinnacle_over_odds   REAL,
            pinnacle_under_odds  REAL,
            total_line           REAL,
            fetched_at           TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(match_date, home_team, away_team)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date   TEXT NOT NULL,
            home_team    TEXT NOT NULL,
            away_team    TEXT NOT NULL,
            home_prob    REAL NOT NULL,
            draw_prob    REAL NOT NULL,
            away_prob    REAL NOT NULL,
            exp_home_goals REAL,
            exp_away_goals REAL,
            created_at   TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(match_date, home_team, away_team)
        );

        CREATE TABLE IF NOT EXISTS soccer_ev_bets (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            match_date            TEXT NOT NULL,
            matchup               TEXT NOT NULL,
            market                TEXT NOT NULL,
            side                  TEXT NOT NULL,
            label                 TEXT NOT NULL,
            model_prob            REAL NOT NULL,
            market_prob           REAL NOT NULL,
            pinnacle_prob         REAL NOT NULL,
            edge_vs_market        REAL NOT NULL,
            entry_odds            REAL NOT NULL,
            entry_book            TEXT,
            ev                    REAL NOT NULL,
            kelly_pct             REAL NOT NULL,
            line_move_direction   INTEGER DEFAULT 0,
            result                TEXT NOT NULL DEFAULT 'TBD',
            created_at            TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(match_date, matchup, market, side)
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date    TEXT NOT NULL,
            n_teams     INTEGER,
            rho         REAL,
            log_lik     REAL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now'))
        );
    """)
    conn.commit()
    conn.close()


if __name__ == "__main__":
    setup_db()
    print(f"DB ready at {DB_PATH}")
