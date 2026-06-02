import os
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv("NHL_DB_PATH", str(Path(__file__).parent / "nhl_predictor.db")))


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.row_factory = sqlite3.Row
    return conn


def setup_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS teams (
            team_code   TEXT PRIMARY KEY,
            team_name   TEXT,
            conference  TEXT,
            division    TEXT
        );

        CREATE TABLE IF NOT EXISTS games (
            game_id     TEXT PRIMARY KEY,
            game_date   TEXT NOT NULL,
            season      TEXT NOT NULL,
            game_type   INTEGER NOT NULL,   -- 2=regular, 3=playoff
            home_team   TEXT NOT NULL,
            away_team   TEXT NOT NULL,
            home_score  INTEGER,
            away_score  INTEGER,
            result      TEXT DEFAULT 'TBD'  -- HOME_WIN, AWAY_WIN, TBD
        );

        CREATE TABLE IF NOT EXISTS team_game_stats (
            game_id             TEXT NOT NULL,
            team_code           TEXT NOT NULL,
            goals               INTEGER,
            shots               INTEGER,
            pp_goals            INTEGER DEFAULT 0,
            pp_opp              INTEGER DEFAULT 0,
            pk_goals_against    INTEGER DEFAULT 0,
            pk_opp              INTEGER DEFAULT 0,
            PRIMARY KEY (game_id, team_code)
        );

        CREATE TABLE IF NOT EXISTS goalie_game_stats (
            game_id         TEXT NOT NULL,
            goalie_id       TEXT NOT NULL,
            team_code       TEXT NOT NULL,
            shots_against   INTEGER,
            saves           INTEGER,
            decision        TEXT,
            PRIMARY KEY (game_id, goalie_id)
        );

        CREATE TABLE IF NOT EXISTS features (
            game_id             TEXT PRIMARY KEY,
            -- Team rolling (last 10 games each side)
            home_gf_10          REAL, away_gf_10          REAL,
            home_ga_10          REAL, away_ga_10          REAL,
            home_sf_10          REAL, away_sf_10          REAL,
            home_sa_10          REAL, away_sa_10          REAL,
            home_shot_pct_10    REAL, away_shot_pct_10    REAL,
            home_sv_pct_10      REAL, away_sv_pct_10      REAL,
            home_pp_pct_10      REAL, away_pp_pct_10      REAL,
            home_pk_pct_10      REAL, away_pk_pct_10      REAL,
            home_win_pct_10     REAL, away_win_pct_10     REAL,
            -- Differentials
            gf_diff             REAL, sf_diff             REAL,
            shot_pct_diff       REAL, sv_pct_diff         REAL,
            pp_pct_diff         REAL, pk_pct_diff         REAL,
            win_pct_diff        REAL,
            -- Goalie rolling (last 5 starts)
            home_goalie_sv_pct_5    REAL, away_goalie_sv_pct_5    REAL,
            home_goalie_gsaa_5      REAL, away_goalie_gsaa_5      REAL,
            goalie_sv_pct_diff      REAL, goalie_gsaa_diff        REAL,
            -- Rest & schedule
            home_rest_days      INTEGER, away_rest_days      INTEGER,
            home_b2b            INTEGER, away_b2b            INTEGER,
            rest_diff           INTEGER,
            -- Elo
            home_elo_prob       REAL, elo_diff             REAL,
            -- Calendar
            is_playoff          INTEGER, month              INTEGER,
            -- PDO and shot share
            home_pdo_10         REAL, away_pdo_10         REAL, pdo_diff          REAL,
            home_shot_share_10  REAL, away_shot_share_10  REAL, shot_share_diff   REAL,
            -- Target
            home_win            INTEGER
        );

        CREATE TABLE IF NOT EXISTS predictions (
            game_id         TEXT PRIMARY KEY,
            prediction_date TEXT,
            home_win_prob   REAL,
            away_win_prob   REAL
        );

        CREATE TABLE IF NOT EXISTS model_runs (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            run_date     TEXT,
            train_end    TEXT,
            val_accuracy REAL,
            val_brier    REAL,
            val_auc      REAL,
            n_train      INTEGER,
            n_val        INTEGER,
            notes        TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_games_date   ON games(game_date);
        CREATE INDEX IF NOT EXISTS idx_games_home   ON games(home_team);
        CREATE INDEX IF NOT EXISTS idx_games_away   ON games(away_team);
        CREATE INDEX IF NOT EXISTS idx_tgs_team     ON team_game_stats(team_code);
        CREATE INDEX IF NOT EXISTS idx_ggs_team     ON goalie_game_stats(team_code);
        CREATE INDEX IF NOT EXISTS idx_ggs_goalie   ON goalie_game_stats(goalie_id);
    """)
    conn.commit()

    # Migrate existing features table to add new columns if absent
    new_feature_cols = [
        ("home_pdo_10",        "REAL"),
        ("away_pdo_10",        "REAL"),
        ("pdo_diff",           "REAL"),
        ("home_shot_share_10", "REAL"),
        ("away_shot_share_10", "REAL"),
        ("shot_share_diff",    "REAL"),
    ]
    for col, dtype in new_feature_cols:
        try:
            conn.execute(f"ALTER TABLE features ADD COLUMN {col} {dtype}")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # column already exists

    conn.close()
    print("Database ready.")


if __name__ == "__main__":
    setup_db()
