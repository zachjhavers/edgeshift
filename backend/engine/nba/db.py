import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("NBA_DB_PATH", str(Path(__file__).parent / "nba_predictor.db")))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id    TEXT PRIMARY KEY,
            game_date  TEXT NOT NULL,
            season     TEXT NOT NULL,
            game_type  TEXT NOT NULL,
            home_team  TEXT NOT NULL,
            away_team  TEXT NOT NULL,
            home_pts   INTEGER,
            home_fgm   INTEGER, home_fga  INTEGER,
            home_fg3m  INTEGER, home_fg3a INTEGER,
            home_ftm   INTEGER, home_fta  INTEGER,
            home_oreb  INTEGER, home_dreb INTEGER,
            home_tov   INTEGER,
            away_pts   INTEGER,
            away_fgm   INTEGER, away_fga  INTEGER,
            away_fg3m  INTEGER, away_fg3a INTEGER,
            away_ftm   INTEGER, away_fta  INTEGER,
            away_oreb  INTEGER, away_dreb INTEGER,
            away_tov   INTEGER,
            home_win   INTEGER
        );

        CREATE TABLE IF NOT EXISTS features (
            game_id         TEXT PRIMARY KEY,
            game_date       TEXT NOT NULL,
            season          TEXT NOT NULL,
            game_type       TEXT NOT NULL,
            home_team       TEXT NOT NULL,
            away_team       TEXT NOT NULL,
            home_efg        REAL, home_tov_pct  REAL, home_orb_pct REAL, home_ftr      REAL,
            home_ortg       REAL, home_drtg     REAL, home_net_rtg REAL, home_pace     REAL,
            away_efg        REAL, away_tov_pct  REAL, away_orb_pct REAL, away_ftr      REAL,
            away_ortg       REAL, away_drtg     REAL, away_net_rtg REAL, away_pace     REAL,
            efg_diff        REAL, tov_pct_diff  REAL, orb_pct_diff REAL, ftr_diff      REAL,
            net_rtg_diff    REAL, pace_avg      REAL,
            home_rest_days  REAL, away_rest_days REAL, rest_diff   REAL,
            home_elo        REAL, away_elo       REAL, elo_diff    REAL,
            home_win        INTEGER
        );

        CREATE TABLE IF NOT EXISTS team_elo (
            team         TEXT PRIMARY KEY,
            elo          REAL NOT NULL DEFAULT 1500.0,
            last_updated TEXT
        );

        CREATE TABLE IF NOT EXISTS historical_odds (
            id                         INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date                  TEXT NOT NULL,
            home_team                  TEXT NOT NULL,
            away_team                  TEXT NOT NULL,
            home_odds                  REAL,
            away_odds                  REAL,
            pinnacle_home_odds         REAL,
            pinnacle_away_odds         REAL,
            best_home_odds             REAL,
            best_away_odds             REAL,
            best_home_book             TEXT,
            best_away_book             TEXT,
            consensus_home_prob        REAL,
            opening_pinnacle_home_prob REAL,
            last_fetched_at            TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(game_date, home_team, away_team)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            game_date     TEXT NOT NULL,
            game_id       TEXT NOT NULL,
            home_team     TEXT NOT NULL,
            away_team     TEXT NOT NULL,
            game_type     TEXT NOT NULL,
            home_win_prob REAL NOT NULL,
            away_win_prob REAL NOT NULL,
            created_at    TEXT NOT NULL,
            UNIQUE(game_date, game_id)
        );

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
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date     TEXT NOT NULL,
            game_type    TEXT NOT NULL,
            val_accuracy REAL,
            val_brier    REAL,
            val_auc      REAL,
            n_train      INTEGER,
            n_val        INTEGER
        );
    """)
    conn.commit()
    conn.close()
